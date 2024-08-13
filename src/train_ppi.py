import argparse
import torch
import torch.optim as optim
from model import PPIClassifier
from main_utils import (load_config, set_seed, get_device, get_run_name, setup_ppi_logs, load_embeddings, concatenate_embeddings, save_model)
from train_utils import (get_ppi_data_loader, ppi_train_step, ppi_test_step)
import os, csv
from datetime import datetime


def main():
    ##################### Setup #####################
    parser = argparse.ArgumentParser(description="Train embedding fusion experiments")
    parser.add_argument("--dataset", choices=['ppi'], required=True, help="Specify which dataset to use for training and evaluation")
    parser.add_argument("--embeddings", type=str, required=True, help="Select pLM embeddings to combine.")
    parser.add_argument("--hidden_dimensions", type=int, help="The size of hidden dimension")
    parser.add_argument("--device", default="cpu", help="Set device")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate metrics on test set")
    args = parser.parse_args()
    
    # Load config, set seed, and get device
    config = load_config(args.dataset)
    set_seed(args.seed)
    device = get_device(args.device)

    run_name = get_run_name(args.dataset, args.embeddings, args.hidden_dimensions, args.seed)

    # Create directories for logging and individual run data
    summary_log_path, individual_file_path = setup_ppi_logs(args.dataset, run_name)

    # Create fused embedding dictionary
    try:
        dictionaries = load_embeddings(args.embeddings, args.dataset)
        embedding_dictionary, input_dimension = concatenate_embeddings(dictionaries)
        hidden_dimension = args.hidden_dimensions
    except ValueError as e:
        print(f"Error: {e}")

    ##################### Model Training #####################
    # Initialize DataLoaders, model, optimizer, and scheduler, depending on which dataset
    model = PPIClassifier(input_dimension=input_dimension, hidden_dimension=hidden_dimension).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['scheduler']['t_0'])
    criterion = torch.nn.BCELoss()

    train_loader = get_ppi_data_loader(embedding_dictionary, config['directories']['train'], config['training']['batch_size'], True)
    validation_loader = get_ppi_data_loader(embedding_dictionary, config['directories']['validation'], config['training']['batch_size'], False)
    test_loader = get_ppi_data_loader(embedding_dictionary, config['directories']['test'], config['training']['batch_size'], False)

    # Create model checkpoint directory
    model_checkpoint_directory = "model_checkpoints"
    os.makedirs(model_checkpoint_directory, exist_ok=True)
    initial_model_save_path = os.path.join(model_checkpoint_directory, f"{run_name}_initial.pth")
    save_model(model, initial_model_save_path)  # Save the initial model state

    # Open files for logging
    with open(summary_log_path, 'a', newline='') as log_file, open(individual_file_path, 'w', newline='') as individual_file:
        log_writer = csv.writer(log_file, delimiter='\t')
        individual_writer = csv.writer(individual_file, delimiter='\t')
        individual_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val AUCROC', 'Val PRC', 'Val Accuracy', 'Val Sensitivity', 'Val Specificity', 'Val Precision', 'Val F1', 'Val MCC'])

        max_metric = float('-inf')
        model_save_path = initial_model_save_path
        
        for epoch in range(1, config['training']['iteration'] + 1):
            train_loss = ppi_train_step(model, train_loader, optimizer, criterion, device)
            val_loss, val_aucroc, val_prc, val_accuracy, val_sensitivity, val_specificity, val_precision, val_f1, val_mcc = ppi_test_step(model, validation_loader, criterion, device)
            scheduler.step()

            individual_writer.writerow([epoch, train_loss, val_loss, val_aucroc, val_prc, val_accuracy, val_sensitivity, val_specificity, val_precision, val_f1, val_mcc])

            watch_metric = val_prc
            if watch_metric > max_metric:
                max_metric = watch_metric
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_save_path = os.path.join(model_checkpoint_directory, f"{run_name}_{timestamp}.pth")
                save_model(model, model_save_path)

        if args.evaluate:
            model.load_state_dict(torch.load(model_save_path))
            best_val_loss, best_val_aucroc, best_val_prc, best_val_accuracy, best_val_sensitivity, best_val_specificity, best_val_precision, best_val_f1, best_val_mcc = ppi_test_step(model, validation_loader, criterion, device)
            test_loss, test_aucroc, test_prc, test_accuracy, test_sensitivity, test_specificity, test_precision, test_f1, test_mcc = ppi_test_step(model, test_loader, criterion, device)
            log_writer.writerow([args.embeddings, hidden_dimension, args.seed, test_aucroc, test_prc, test_accuracy, test_sensitivity, test_specificity, test_precision, test_f1, test_mcc, best_val_aucroc, best_val_prc, best_val_accuracy, best_val_sensitivity, best_val_specificity, best_val_precision, best_val_f1, best_val_mcc, best_val_loss])



if __name__ == "__main__":
    main()
