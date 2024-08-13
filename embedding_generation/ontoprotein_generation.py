import torch
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re

def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("zjunlp/OntoProtein")
    model = AutoModelForMaskedLM.from_pretrained("zjunlp/OntoProtein")
    model.to(device)
    return model, tokenizer

def get_protein_embeddings(model, tokenizer, sequence, device):
    encoded_input = tokenizer(sequence, return_tensors='pt')
    encoded_input.to(device)

    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
    
    embeddings = output.hidden_states[-1][:, 1:-1, :].squeeze().to('cpu')
    return torch.mean(embeddings, dim=0)

def process_sequences(model, tokenizer, data_list, max_length, device, protein_dictionary):
    cpu_queue = []

    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")

        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        
        try:
            protein_dictionary[uniprot_id] = get_protein_embeddings(model, tokenizer, sequence, device)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error; queuing {uniprot_id} for CPU processing.")
                cpu_queue.append((uniprot_id, sequence))
            else:
                raise e

    if cpu_queue:
        model.to('cpu')

        for uniprot_id, sequence in cpu_queue:
            protein_dictionary[uniprot_id] = get_protein_embeddings(model, tokenizer, sequence, 'cpu')

def main():
    datasets = ['aav_sequences']
    max_length = 5800
    base_dir = "../../Benchmarking-tpLMs/data/aav"
    output_dir = '../../Benchmarking-tpLMs/embeddings/aav/ontoprotein'
    protein_dictionary = {}
    combined_data_list = []

    device = torch.device('cuda')
    model, tokenizer = load_model_and_tokenizer(device)
    model.eval()

    for dataset in datasets:
        data_file = os.path.join(base_dir, f"{dataset}.tsv")
        with open(data_file, "r") as f:
            combined_data_list.extend(f.read().strip().split('\n'))

    process_sequences(model, tokenizer, combined_data_list, max_length, device, protein_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))

if __name__ == "__main__":
    main()
