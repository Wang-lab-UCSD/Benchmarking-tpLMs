import argparse
import os, csv, uuid
import torch
import torch.optim as optim
from model import PPIClassifier
from main_utils import (load_config, set_seed, get_device, get_run_name, setup_ppi_logs, load_embeddings, concatenate_embeddings, save_model)
from train_utils import (get_ppi_data_loader, ppi_train_step, ppi_test_step)

def train(config, args, model, embedding_dict, device):
    run_name = get_run_name(args.dataset, args.embeddings, args.hidden_dimensions, args.seed)
    summary_log_path, individual_file_path = setup_ppi_logs(args.dataset, run_name)

    train_loader = get_ppi_data_loader(embedding_dict, config['directories']['train'], config['training']['batch_size'], True)
    val_loader = get_ppi_data_loader(embedding_dict, config['directories']['validation'], config['training']['batch_size'], False)
    test_loader = get_ppi_data_loader(embedding_dict, config['directories']['test'], config['training']['batch_size'], False)

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['scheduler']['t_0'])
    criterion = torch.nn.BCELoss()

    checkpoint_dir = os.path.join("model_checkpoints", args.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(checkpoint_dir, f"{run_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    best_model_save_path = os.path.join(run_dir, "best_model.pth")

    max_metric = float('-inf')
    
    with open(summary_log_path, 'a', newline='') as log_file, open(individual_file_path, 'w', newline='') as ind_file:
        log_writer = csv.writer(log_file, delimiter='\t')
        ind_writer = csv.writer(ind_file, delimiter='\t')
        ind_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val AUCROC', 'Val PRC', 'Val Accuracy', 'Val Sensitivity', 'Val Specificity', 'Val Precision', 'Val F1', 'Val MCC'])

        for epoch in range(1, config['training']['iteration'] + 1):
            train_loss = ppi_train_step(model, train_loader, optimizer, criterion, device)
            val_loss, val_aucroc, val_prc, val_accuracy, val_sensitivity, val_specificity, val_precision, val_f1, val_mcc = ppi_test_step(model, val_loader, criterion, device)
            scheduler.step()
            ind_writer.writerow([epoch, train_loss, val_loss, val_aucroc, val_prc, val_accuracy, val_sensitivity, val_specificity, val_precision, val_f1, val_mcc])

            if val_prc > max_metric:
                max_metric = val_prc
                save_model(model, best_model_save_path)

        if args.evaluate:
            model.load_state_dict(torch.load(best_model_save_path))
            best_val_loss, best_val_aucroc, best_val_prc, best_val_accuracy, best_val_sensitivity, best_val_specificity, best_val_precision, best_val_f1, best_val_mcc = ppi_test_step(model, val_loader, criterion, device)
            test_loss, test_aucroc, test_prc, test_accuracy, test_sensitivity, test_specificity, test_precision, test_f1, test_mcc = ppi_test_step(model, test_loader, criterion, device)
            log_writer.writerow([args.embeddings, args.hidden_dimensions, args.seed, test_aucroc, test_prc, test_accuracy, test_sensitivity, test_specificity, test_precision, test_f1, test_mcc, best_val_aucroc, best_val_prc, best_val_accuracy, best_val_sensitivity, best_val_specificity, best_val_precision, best_val_f1, best_val_mcc, best_val_loss])


def predict(config, args, model, embedding_dict, device):
    try:
        checkpoint_path = os.path.join("model_checkpoints", args.model_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    test_loader = get_ppi_data_loader(embedding_dict, config['directories']['test'], config['training']['batch_size'], False)
    criterion = torch.nn.BCELoss()

    _, _, _, _, _, _, _, _, _, all_probs = ppi_test_step(model, test_loader, criterion, device, return_preds=True)

    model_dir = os.path.dirname(args.model_path)
    model_id = os.path.basename(model_dir)  # e.g., 'ppi_1_1024_2_f0e2de08'

    try:
        dataset, embedding, hidden_dim, seed, uid = model_id.split('_')
    except ValueError:
        dataset = 'unknown'
        embedding = 'unknown'
        model_id = model_id.replace('/', '_')

    output_dir = os.path.join('results', 'ppi_results', 'predicted_probs')
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename like: ppi_1_1024_2.tsv or ppi_B_1024_2.tsv
    filename = f"{dataset}_{embedding}_{hidden_dim}_{seed}.tsv"
    output_path = os.path.join(output_dir, filename)

    # Write probabilities
    with open(output_path, 'w') as f:
        for prob in all_probs:
            f.write(f"{prob}\n")
def main():
    parser = argparse.ArgumentParser(description="Run PPI training or inference")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--dataset", choices=['ppi'], required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--hidden_dimensions", type=int, required=True)
    parser.add_argument("--model_path", type=str, help="Path to .pth (inference only)")
    parser.add_argument("--output_dir", type=str, help="Where to save inference results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate metrics on test set")

    args = parser.parse_args()
    config = load_config(args.dataset)
    set_seed(args.seed)
    device = get_device(args.device)

    try:
        dicts = load_embeddings(args.embeddings, args.dataset)
        embedding_dict, input_dim = concatenate_embeddings(dicts)
    except ValueError as e:
        print(f"Error loading embeddings: {e}")
        return

    model = PPIClassifier(input_dimension=input_dim, hidden_dimension=args.hidden_dimensions).to(device)

    if args.mode == "train":
        train(config, args, model, embedding_dict, device)
    else:
        predict(config, args, model, embedding_dict, device)

if __name__ == "__main__":
    main()
