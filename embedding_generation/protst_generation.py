import torch
from transformers import BertTokenizer, BertForMaskedLM
from protst.model import *
from protst.task import ProtSTMMP
from protst.data import PackedProtein
import os


def load_model(device):
    text_model = PubMedBERT(model='PubMedBERT-abs')
    protein_model = PretrainEvolutionaryScaleModeling(path='~/scratch/esm-model-weights/', readout='mean',
                                                    model='ESM-2-650M', mask_modeling=True,output_dim=512, num_mlp_layer=2, activation='relu',use_proj=True)
    fusion_model = CrossAttention(batch_norm=True)
    model = ProtSTMMP(protein_model, text_model, fusion_model, protein2text=True, text2protein=True, mlm_weight=1.0, mmp_weight=1.0, global_contrast=True)
    model.to(device)
    checkpoint_path = '/new-stg/home/young/embedding_fusion/ProtST/protst_esm2.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'], strict=False)

    return model.protein_model

def sanitize(sequence):
    replacements = {
        'U': 'C', 'B': 'N', 'Z': 'Q', 'J': 'L', 'O': 'C', 'X': ''
    }
    cleaned = []
    for aa in sequence:
        if aa in replacements:
            cleaned.append(replacements[aa])
        else:
            cleaned.append(aa)
    return ''.join(cleaned)

def get_protein_embeddings(model, sequence, device):

    protein_graph = PackedProtein.from_sequence(sequence)
    protein_graph.to(device)
    with torch.no_grad():
        dictionary = model(protein_graph, sequence)
        return dictionary['graph_feature'].squeeze().to('cpu'), torch.mean(dictionary['residue_feature'], dim=0).to('cpu')
        
def process_sequences(model, data_list, max_length, device, averaged_protein_dictionary, residue_level_dictionary):
    cpu_queue = []
    
    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")
        
        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]
        sequence = sanitize(sequence)
        
        try:
            # per protein, per residue embeddings
            averaged_protein_dictionary[uniprot_id], residue_level_dictionary[uniprot_id] = get_protein_embeddings(model, [sequence], device)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error; queuing {uniprot_id} for CPU processing.")
                cpu_queue.append((uniprot_id, sequence))
    
    model.to('cpu')
    for uniprot_id, sequence in cpu_queue:
        averaged_protein_dictionary[uniprot_id], residue_level_dictionary[uniprot_id] = get_protein_embeddings(model, [sequence], device)
    
if __name__ == "__main__":

    datasets = ['cath_sequences']
    max_length = 5800
    base_dir = "../../Benchmarking-tpLMs/data/cath"
    output_dir = '../../Benchmarking-tpLMs/embeddings/cath/protst'
    averaged_protein_dictionary = {}
    residue_level_dictionary = {}
    
    combined_data_list = []
    device = torch.device('cuda')
    model = load_model(device)
    model.eval()

    for dataset in datasets:
        data_file = os.path.join(base_dir, f"{dataset}.tsv")
        with open(data_file, "r") as f:
            combined_data_list.extend(f.read().strip().split('\n'))

    process_sequences(model, combined_data_list, max_length, device, averaged_protein_dictionary, residue_level_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(averaged_protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))
