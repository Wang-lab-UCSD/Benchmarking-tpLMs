import torch
import torch.nn.functional as F
import yaml
import random
import numpy as np
import os, csv

# ------------------- Initialization and Configuration -------------------
# Open and load the config file
def load_config(dataset):
    config_file_path = os.path.join("configs", f"{dataset}.yaml")
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Use gpu if available
def get_device(cuda_device=None):
    if cuda_device:
        return torch.device(cuda_device)
    else:
        return torch.device('cpu')

# Create run name for logging purposes
def get_run_name(dataset, embedding_string, hidden_dimension, seed):
    run_name = f"{dataset}_{embedding_string}_{str(hidden_dimension)}_{str(seed)}"
    return run_name

# Set up logging files
def setup_logging_directories(dataset, run_name, embedding_string):
    if len(embedding_string) == 1:
        log_directory = os.path.join("results", "tplm_benchmark_results")
        os.makedirs(log_directory, exist_ok=True)
    else:
        log_directory = os.path.join("results", "embedding_fusion_results")
        os.makedirs(log_directory, exist_ok=True)
    
    log_path = os.path.join(log_directory, f'{dataset}_log.tsv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as log_file:
            log_writer = csv.writer(log_file, delimiter='\t')
            log_writer.writerow(['Dataset', 'Embedding', 'Hidden Dimension', 'Seed', 'Best Val Metric', 'Test Metric', 'Parameter Count'])

    individual_runs_dir = os.path.join(log_directory, "individual_runs", dataset)
    os.makedirs(individual_runs_dir, exist_ok=True)
    individual_file_path = os.path.join(individual_runs_dir, f"{run_name}.tsv")

    return log_path, individual_file_path

def setup_ppi_logs(dataset, run_name):
    log_directory = os.path.join("results", "ppi_results")
    os.makedirs(log_directory, exist_ok=True)

    ppi_log_path = os.path.join(log_directory, 'ppi_log.tsv')
    if not os.path.exists(ppi_log_path):
        with open(ppi_log_path, 'w', newline='') as log_file:
            log_writer = csv.writer(log_file, delimiter='\t')
            log_writer.writerow(['Embedding', 'Hidden Dimension', 'Seed', 'Test AUROC', 'Test PRC', 'Test Accuracy', 'Test Sensitivity', 'Test Specificity', 'Test Precision', 'Test F1', 'Test MCC', 'Best Val AUROC', 'Best Val PRC', 'Best Val Accuracy', 'Best Val Sensitivity', 'Best Val Specificity', 'Best Val Precision', 'Best Val F1', 'Best Val MCC', 'Parameter Count'])
    
    individual_runs_dir = os.path.join(log_directory, "individual_runs", dataset)
    os.makedirs(individual_runs_dir, exist_ok=True)
    individual_file_path = os.path.join(individual_runs_dir, f"{run_name}.tsv")

    return ppi_log_path, individual_file_path

def setup_cath_logs():
    log_directory = os.path.join("results", "cath_results")
    os.makedirs(log_directory, exist_ok=True)

    cath_log_path = os.path.join(log_directory, 'cath_log.tsv')
    if not os.path.exists(cath_log_path):
        with open(cath_log_path, 'w', newline='') as log_file:
            log_writer = csv.writer(log_file, delimiter='\t')
            log_writer.writerow(['Embedding', 'Accuracy'])

    return cath_log_path

# Load embedding dictionary
def load_embeddings(embedding_string, task):
    task_directories = {
        'aav': 'aav',
        'gb1': 'gb1',
        'gfp': 'gfp',
        'location': 'location',
        'meltome': 'meltome',
        'stability': 'stability',
        'ppi': 'protein-protein',
        'cath':'cath'
    }

    if task not in task_directories:
        raise ValueError(f"Unknown task '{task}'. Available tasks are: {list(task_directories.keys())}")

    base_directory = 'embeddings'
    task_directory = os.path.join(base_directory, task_directories[task])

    embedding_map = {
        'A': os.path.join(task_directory, 'esm2(3B)/protein_dictionary.pt'),
        'B': os.path.join(task_directory, 'esm3/protein_dictionary.pt'),
        'C': os.path.join(task_directory, 'ontoprotein/protein_dictionary.pt'),
        'D': os.path.join(task_directory, 'proteinclip_esm3b/protein_dictionary.pt') if task=='cath' else os.path.join(task_directory, 'proteinclip_t5/protein_dictionary.pt') ,
        'E': os.path.join(task_directory, 'protst/protein_dictionary.pt'), 
        'F': os.path.join(task_directory, 'protrek/protein_dictionary.pt'),
        'G': os.path.join(task_directory, 'proteindt/protein_dictionary.pt'),
    }

    dictionaries = []
    for char in embedding_string:
        if char in embedding_map:
            try:
                dictionary = torch.load(embedding_map[char])
                dictionaries.append(dictionary)
            except Exception as e: 
                raise ValueError(f"Failed to load embeddings for '{char}' at '{embedding_map[char]}': {str(e)}")
        else:
            raise ValueError(f"Invalid character '{char}' in embedding string. Valid keys are: {list(embedding_map.keys())}")
    return dictionaries

# Given embeddings [N], [M], [K],  return [N+M+K], concatenated in order of input
def concatenate_embeddings(dict_list):
    if not dict_list:
        raise ValueError("No embedding dictionaries provided.")

    concatenated_embedding_dict = {}
    keys = dict_list[0].keys()

    for key in keys:
        tensors_to_concat = [d[key] for d in dict_list if key in d]
        if not tensors_to_concat:
            raise ValueError(f"No embeddings found for key '{key}'.")
        concatenated_embedding_dict[key] = torch.cat(tensors_to_concat, dim=0)

    if concatenated_embedding_dict:
        one_key = next(iter(concatenated_embedding_dict))
        embedding_dim = concatenated_embedding_dict[one_key].shape[0]
    else:
        raise ValueError("No valid embeddings to concatenate.")

    return concatenated_embedding_dict, embedding_dim

# ------------------- Model utils -------------------
# Return number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Saving best model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


# ------------------- Extra Utils for CATH pipeline -------------------
def load_superfamily_mapping(tsv_file):
    superfamily_dict = {}
    with open(tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            protein_id, superfamily_id = row
            superfamily_dict[protein_id] = superfamily_id
    return superfamily_dict

def normalize_embeddings(embedding_dict, device):
    for key, embedding in embedding_dict.items():
        norm_embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)  # Apply L2 norm
        embedding_dict[key] = norm_embedding.squeeze(0).to(device)
    return embedding_dict