import torch
import torch.nn as nn
import torch.nn.functional as F

# Model architecture based on https://www.biorxiv.org/content/10.1101/2023.12.13.571462v1 (https://github.com/RSchmirler/data-repo_plm-finetune-eval/blob/main/notebooks/embedding/Embedding_Predictor_Training.ipynb)
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension, dropout_rate):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.dropout_rate = dropout_rate

        self.normalize = nn.BatchNorm1d(self.input_dimension)
        self.fully_connected = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dimension, self.output_dimension)
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fully_connected.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, protein):
        
        projected_protein = self.dropout(self.relu(self.fully_connected(self.normalize(protein))))

        predicted_fitness = self.output_layer(projected_protein)

        return predicted_fitness.squeeze()

# Model architecture based on https://www.biorxiv.org/content/10.1101/2024.01.29.577794v1 (https://github.com/navid-naderi/PLM_SWE/tree/main)
class PPIClassifier(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension

        self.fully_connected = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fully_connected.weight)
        
    def forward(self, proteinA, proteinB):
        
        combined_proteins = torch.cat((proteinA, proteinB), dim=0)
        projected_combined = self.relu(self.fully_connected(combined_proteins))
        num_pairs  = len(combined_proteins) // 2
        cosine_similarity = F.cosine_similarity(projected_combined[:num_pairs], projected_combined[num_pairs:])
        
        return cosine_similarity
