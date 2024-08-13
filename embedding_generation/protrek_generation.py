import torch
import os
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

def load_model(device):
    config = {
        "protein_config": "weights/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
        "text_config": "weights/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": "weights/ProTrek_650M_UniRef50/foldseek_t30_150M",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": "weights/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
    }

    model = ProTrekTrimodalModel(**config).eval().to(device)
    return model

def get_protein_embeddings(model, sequence):

    with torch.no_grad():
        results = model.get_protein_repr([sequence])
        return results.squeeze().to('cpu')

def process_sequences(model, data_list, max_length, device, protein_dictionary):
    cpu_queue = []

    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")

        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]

        try:
            protein_dictionary[uniprot_id] = get_protein_embeddings(model, sequence)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error; queuing {uniprot_id} for CPU processing.")
                cpu_queue.append((uniprot_id, sequence))
            else:
                raise e

    if cpu_queue:
        model.to('cpu')

        for uniprot_id, sequence in cpu_queue:
            protein_dictionary[uniprot_id] = get_protein_embeddings(model, sequence)

def main():
    datasets = ['stability_sequences']
    max_length = 5800
    base_dir = "../../Benchmarking-tpLMs/data/stability"
    output_dir = '../../Benchmarking-tpLMs/embeddings/stability/protrek'
    protein_dictionary = {}
    combined_data_list = []

    device = torch.device('cuda')
    model= load_model(device)

    for dataset in datasets:
        data_file = os.path.join(base_dir, f"{dataset}.tsv")
        with open(data_file, "r") as f:
            combined_data_list.extend(f.read().strip().split('\n'))

    process_sequences(model, combined_data_list, max_length, device, protein_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))

if __name__ == "__main__":
    main()
