import argparse
import torch
import torch.optim as optim
from model import EmbeddingNetwork, WideEmbeddingNetwork
from main_utils import (
    load_config, set_seed, get_device, get_run_name,
    setup_logging_directories, load_embeddings, concatenate_embeddings,
    count_parameters, save_model
)
from train_utils import (
    get_data_loader, train_step, test_step
)
import os, csv, uuid, time

def train(args, config, model, embedding_dict, device, log_path, checkpoint_path):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], eps=1e-7)
    criterion = torch.nn.CrossEntropyLoss() if args.dataset == 'location' else torch.nn.MSELoss()

    train_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['train'], config['training']['batch_size'], True)
    val_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['validation'], config['training']['batch_size'], False)
    test_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['test'], config['training']['batch_size'], False)

    best_val_loss = float('inf')

    with open(log_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter='\t')

        for epoch in range(1, config['training']['iteration'] + 1):
            train_loss = train_step(model, train_loader, optimizer, criterion, device)
            val_loss, val_metric = test_step(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, checkpoint_path)

        if args.evaluate:
            model.load_state_dict(torch.load(checkpoint_path))
            _, val_metric = test_step(model, val_loader, criterion, device)
            _, test_metric = test_step(model, test_loader, criterion, device)

            param_count = count_parameters(model)
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            run_time = time.time() - args.start_time

            log_writer.writerow([args.dataset, args.embeddings, args.hidden_dimension, args.seed, val_metric, test_metric, param_count, max_mem, run_time])

def inference(args, config, model, embedding_dict, device):
    test_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['test'], config['training']['batch_size'], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss() if args.dataset == 'location' else torch.nn.MSELoss()

    checkpoint_path = args.model_path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, _, all_probs = test_step(model, test_loader, criterion, device, return_preds=True)

    model_id = os.path.basename(os.path.dirname(args.model_path))
    output_path = os.path.join('results','predicted_probs', args.dataset, f"{model_id}.tsv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for prob in all_probs:
            f.write(f"{prob}\n")

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on pLM tasks")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--dataset", choices=['aav', 'gb1', 'gfp', 'location', 'meltome', 'stability'], required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--hidden_dimension", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--model_path", type=str, help="Path to saved model for inference")

    args = parser.parse_args()
    args.start_time = time.time()
    set_seed(args.seed)
    config = load_config(args.dataset)
    device = get_device(args.device)

    run_name = get_run_name(args.dataset, args.embeddings, args.hidden_dimension, args.seed)
    checkpoint_dir = os.path.join("model_checkpoints", args.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = setup_logging_directories(args.dataset, run_name, args.embeddings)

    try:
        dicts = load_embeddings(args.embeddings, args.dataset)
        embedding_dict, input_dim = concatenate_embeddings(dicts)
    except ValueError as e:
        print(f"Error loading embeddings: {e}")
        return

    model = EmbeddingNetwork(input_dimension=input_dim, output_dimension=10 if args.dataset=='location' else 1, hidden_dimension=args.hidden_dimension, dropout_rate=args.dropout).to(device)
    model_dir = f"{run_name}_{uuid.uuid4().hex[:8]}"
    checkpoint_path = os.path.join(checkpoint_dir, model_dir, "best_model.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if args.mode == "train":
        train(args, config, model, embedding_dict, device, log_path, checkpoint_path)
    else:
        inference(args, config, model, embedding_dict, device)

if __name__ == "__main__":
    main()
