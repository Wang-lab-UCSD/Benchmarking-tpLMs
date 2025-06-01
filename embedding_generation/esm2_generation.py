import torch
import os
import esm

def load_model_and_alphabet(device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    model.eval()
    return model, alphabet

def get_protein_embeddings(model, alphabet, sequence, device):
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        return torch.mean(results["representations"][33][0, 1:-1, :].to('cpu'), dim=0)

def process_sequences(model, alphabet, data_list, max_length, device, protein_dictionary):
    cpu_queue = []

    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")

        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]

        try:
            protein_dictionary[uniprot_id] = get_protein_embeddings(model, alphabet, sequence, device)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error; queuing {uniprot_id} for CPU processing.")
                cpu_queue.append((uniprot_id, sequence))
            else:
                raise e

    if cpu_queue:
        model.to('cpu')

        for uniprot_id, sequence in cpu_queue:
            protein_dictionary[uniprot_id] = get_protein_embeddings(model, alphabet, sequence, 'cpu')

def main():
    #datasets = ['aav', 'gb1', 'gfp', 'location', 'meltome', 'stability']
    datasets = ['ppi']
    max_length = 5800

    device = torch.device('cuda')
    model, alphabet = load_model_and_alphabet(device)
    model.eval()

    for dataset in datasets:
        base_dir = f"../../Benchmarking-tpLMs/data/{dataset}"
        output_dir = f'../../Benchmarking-tpLMs/embeddings/{dataset}/esm2(650M)'
        protein_dictionary = {}
        data_list = []
        data_file = os.path.join(base_dir, f"{dataset}_sequences.tsv")
        with open(data_file, "r") as f:
            data_list.extend(f.read().strip().split('\n'))
        process_sequences(model, alphabet, data_list, max_length, device, protein_dictionary)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))

if __name__ == "__main__":
    main()
