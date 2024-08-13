import torch
from transformers import T5Tokenizer, T5EncoderModel
from proteinclip import model_utils
import os
import re

def load_model_and_tokenizer(device):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    base_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    clip_model = model_utils.load_proteinclip("t5")
    base_model.to(device)
    return base_model, clip_model, tokenizer

def get_protein_embeddings(base_model, clip_model, tokenizer, sequence, device):
    
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence)))]

    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding_repr = base_model(input_ids=input_ids,attention_mask=attention_mask)
        base_embedding = torch.mean(embedding_repr.last_hidden_state[0,:-1], dim=0).to('cpu')
        clip_embedding = clip_model.predict(base_embedding.numpy())

        return torch.from_numpy(clip_embedding)

def process_sequences(base_model, clip_model, tokenizer, data_list, max_length, device, protein_dictionary):
    cpu_queue = []

    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")

        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]

        try:
            protein_dictionary[uniprot_id] = get_protein_embeddings(base_model, clip_model, tokenizer, sequence, device)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error; queuing {uniprot_id} for CPU processing.")
                cpu_queue.append((uniprot_id, sequence))
            else:
                raise e

    if cpu_queue:
        base_model.to('cpu')

        for uniprot_id, sequence in cpu_queue:
            protein_dictionary[uniprot_id] = get_protein_embeddings(base_model, clip_model, tokenizer, sequence, 'cpu')

def main():
    datasets = ['meltome_sequences']
    max_length = 5800
    base_dir = "../../Benchmarking-tpLMs/data/meltome"
    output_dir = '../../Benchmarking-tpLMs/embeddings/meltome/proteinclip_t5'
    protein_dictionary = {}
    combined_data_list = []

    device = torch.device('cuda')
    base_model, clip_model, tokenizer = load_model_and_tokenizer(device)
    base_model.eval()

    for dataset in datasets:
        data_file = os.path.join(base_dir, f"{dataset}.tsv")
        with open(data_file, "r") as f:
            combined_data_list.extend(f.read().strip().split('\n'))

    process_sequences(base_model, clip_model, tokenizer, combined_data_list, max_length, device, protein_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))

if __name__ == "__main__":
    main()
