import torch
import os
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import torch._dynamo
#torch._dynamo.config.suppress_errors = True

def load_client(device):
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
    return client

def get_protein_embeddings(client, sequence):
    protein = ESMProtein(sequence=(sequence))
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(
        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
    )

    return torch.mean(output.per_residue_embedding[1:-1], dim=0).squeeze().to('cpu')

def process_sequences(client, data_list, max_length, protein_dictionary):
    cpu_queue = []

    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")

        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]

        try:
            protein_dictionary[uniprot_id] = get_protein_embeddings(client, sequence)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error; queuing {uniprot_id} for CPU processing.")
                cpu_queue.append((uniprot_id, sequence))
            else:
                raise e

    if cpu_queue:
        client.to('cpu')

        for uniprot_id, sequence in cpu_queue:
            protein_dictionary[uniprot_id] = get_protein_embeddings(client, sequence)

def main():
    datasets = ['location_sequences']
    max_length = 5800
    base_dir = "../../Benchmarking-tpLMs/data/location"
    output_dir = '../../Benchmarking-tpLMs/embeddings/location/esm3'
    protein_dictionary = {}
    combined_data_list = []

    device = torch.device('cuda')
    client = load_client(device)

    for dataset in datasets:
        data_file = os.path.join(base_dir, f"{dataset}.tsv")
        with open(data_file, "r") as f:
            combined_data_list.extend(f.read().strip().split('\n'))

    process_sequences(client, combined_data_list, max_length, protein_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))

if __name__ == "__main__":
    main()
