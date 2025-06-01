import argparse
import torch
import torch.optim as optim
from model import EmbeddingNetwork
from main_utils import (load_config, set_seed, get_device, get_run_name, setup_logging_directories, load_embeddings, concatenate_embeddings, count_parameters, save_model)
from train_utils import (get_data_loader, train_step, test_step)
import os, csv
import uuid
import time


def main():
    ##################### Setup #####################
    parser = argparse.ArgumentParser(description="Train models for benchmarking")
    parser.add_argument("--dataset", choices=['aav', 'gb1', 'gfp', 'location', 'meltome', 'stability'], required=True, help="Specify which dataset to use for training and evaluation")
    parser.add_argument("--embeddings", type=str, required=True, help="Specify which pLM embeddings to use.")
    parser.add_argument("--hidden_dimension", type=int, help="The size of hidden dimension")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate for the model")
    parser.add_argument("--device", default="cpu", help="Set device")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate metrics on test set")
    args = parser.parse_args()
    
    # Load config, set seed, and get device
    start_time = time.time()
    config = load_config(args.dataset)
    set_seed(args.seed)
    device = get_device(args.device)
    torch.cuda.reset_peak_memory_stats(device)

    run_name = get_run_name(args.dataset, args.embeddings, args.hidden_dimension, args.seed)

    # Create directories for logging the test results as well per epoch train/val results
    summary_log_path  = setup_logging_directories(args.dataset, run_name, args.embeddings)

    # Create fused embedding dictionary (also works even if theres only one embedding specified)
    try:
        dictionaries = load_embeddings(args.embeddings, args.dataset)
        embedding_dictionary, input_dimension = concatenate_embeddings(dictionaries)
    
    except ValueError as e:
        print(f"Error: {e}")

    ##################### Model Training #####################
    # Initialize model and optimizer
    model = EmbeddingNetwork(input_dimension=input_dimension, output_dimension=10 if args.dataset=='location' else 1, hidden_dimension=args.hidden_dimension, dropout_rate=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], eps=1e-7)
    criterion = torch.nn.CrossEntropyLoss() if args.dataset=='location' else torch.nn.MSELoss()
    parameter_count = count_parameters(model)

    # Initialize data loaders
    train_loader = get_data_loader(args.dataset, embedding_dictionary, config['directories']['train'], config['training']['batch_size'], True)
    validation_loader = get_data_loader(args.dataset, embedding_dictionary, config['directories']['validation'], config['training']['batch_size'], False)
    test_loader = get_data_loader(args.dataset, embedding_dictionary, config['directories']['test'], config['training']['batch_size'], False)

    # Create model checkpoint directory
    model_checkpoint_directory = f"model_checkpoints/{args.dataset}"
    os.makedirs(model_checkpoint_directory, exist_ok=True)
    
    # Create a unique subfolder for each run to avoid name collisions
    unique_id = uuid.uuid4().hex[:8]
    unique_run_name = f"{run_name}_{unique_id}"
    model_checkpoint_run_directory = os.path.join(model_checkpoint_directory, unique_run_name)
    os.makedirs(model_checkpoint_run_directory, exist_ok=True)
    best_model_save_path = os.path.join(model_checkpoint_run_directory, "best_model.pth")

    # Open files for logging
    with open(summary_log_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter='\t')


        max_metric = float('inf')
        
        for epoch in range(1, config['training']['iteration'] + 1):
            train_loss = train_step(model, train_loader, optimizer, criterion, device)
            val_loss, val_metric = test_step(model, validation_loader, criterion, device)

            if val_loss < max_metric:
                max_metric = val_loss
                save_model(model, best_model_save_path)
        
        # Record the training run time
        max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)  # convert to MB
        end_time = time.time()
        run_time = end_time - start_time

        if args.evaluate:
            # Load the model with lowest validation loss for evaluation on the test set
            model.load_state_dict(torch.load(best_model_save_path))
            best_val_loss, best_val_metric = test_step(model, validation_loader, criterion, device)
            test_loss, test_metric = test_step(model, test_loader, criterion, device)
            log_writer.writerow([args.dataset, args.embeddings, args.hidden_dimension, args.seed, best_val_metric, test_metric, parameter_count, max_mem, run_time])

if __name__ == "__main__":
    main()
