import argparse
import torch
from model import PPIClassifier
from main_utils import (load_config, set_seed, get_device, load_embeddings, concatenate_embeddings)
from train_utils import (get_ppi_data_loader, ppi_test_step)
import os, csv
import pandas as pd

def main():
    ##################### Setup #####################
    parser = argparse.ArgumentParser(description="Run inference with a trained PPI classifier model")
    parser.add_argument("--dataset", choices=['ppi'], required=True, help="Specify which dataset to use for training and evaluation")
    parser.add_argument("--embeddings", type=str, required=True, help="Select pLM embeddings to combine")
    parser.add_argument("--hidden_dimensions", type=int, default=1024, required=True, help="The size of hidden dimension (must match the trained model)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument("--output_dir", type=str, help="Path to save inference results")
    parser.add_argument("--device", default="cpu", help="Set device")
    args = parser.parse_args()
    
    # Load config, set seed, and get device
    config = load_config(args.dataset)
    device = get_device(args.device)
    
    # Load embeddings
    try:
        dictionaries = load_embeddings(args.embeddings, args.dataset)
        embedding_dictionary, input_dimension = concatenate_embeddings(dictionaries)
        hidden_dimension = args.hidden_dimensions
    except ValueError as e:
        print(f"Error loading embeddings: {e}")
        return

    ##################### Model Inference #####################
    # Initialize model and load trained weights
    model = PPIClassifier(input_dimension=input_dimension, hidden_dimension=hidden_dimension).to(device)
    
    try:
        checkpoint_path = 'model_checkpoints/'
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, args.model_path), map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    
    # Create test data loader
    test_loader = get_ppi_data_loader(embedding_dictionary, config['directories']['test'], config['training']['batch_size'], False)
    
    # Run inference on test set
    criterion = torch.nn.BCELoss()
    test_loss, test_aucroc, test_prc, test_accuracy, test_sensitivity, test_specificity, test_precision, test_f1, test_mcc, all_probabilities = ppi_test_step(model, test_loader, criterion, device, return_preds=True)
    output_dir = 'results/ppi_results/predicted_probs'
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    prob_output_file = os.path.join(output_dir, f"{model_name}.tsv")
    # Save the probabilities (one per line)
    with open(prob_output_file, 'w') as f:
        for prob in all_probabilities:
            f.write(f"{prob}\n")

if __name__ == "__main__":
    main()
