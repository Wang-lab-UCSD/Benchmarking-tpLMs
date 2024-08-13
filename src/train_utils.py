import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, confusion_matrix, precision_recall_curve, precision_score, recall_score, confusion_matrix, f1_score, matthews_corrcoef
import numpy as np
import scipy.stats

############ Utils for AAV, GB1, GFP, Location, Meltome, Stability Dataset Models ############
class BenchmarkDataset(Dataset):
    def __init__(self, dictionary, file_path, target_type=float):
        self.dictionary = dictionary
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                protein, target = line.strip().split("\t")
                if target_type == float:
                    target_value = float(target)
                    target_tensor = torch.tensor(target_value, dtype=torch.float32)
                else:
                    target_value = int(target)
                    target_tensor = torch.tensor(target_value, dtype=torch.long)
                self.data.append((protein, target_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        protein, target = self.data[index]
        return self.dictionary[protein], target
    
def get_data_loader(dataset_type, dictionary, action_file, batch_size, shuffle):

    dataset = BenchmarkDataset(dictionary, action_file, target_type=int if dataset_type == 'location' else float)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_step(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for protein, target in data_loader:
        optimizer.zero_grad()
        protein, target = protein.to(device), target.to(device)
        predictions = model(protein)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

        batch_size = len(protein)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
    
    return total_loss / total_samples

def test_step(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for protein, target in data_loader:
            protein, target = protein.to(device), target.to(device)
            predictions = model(protein)
            loss = criterion(predictions, target)

            batch_size = len(protein)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            all_targets.append(target)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                all_predictions.append(predictions.argmax(dim=1))  # If we are doing Location Classification
            else:
                all_predictions.append(predictions)  # If we are doing any other dataset Regression

    all_targets = torch.cat(all_targets, dim=0).detach().cpu().numpy()
    all_predictions = torch.cat(all_predictions, dim=0).detach().cpu().numpy()

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        metric = calculate_accuracy(all_targets, all_predictions)
    else:
        metric = calculate_spearman(all_targets, all_predictions)

    return total_loss / total_samples, metric

def calculate_spearman(y_true, y_pred):
    return scipy.stats.spearmanr(y_true, y_pred).correlation

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

############ Protein-protein interaction classification utils ############
class ProteinInteractionDataset(Dataset):
    def __init__(self, protein_dict, protein_interactions_file):
        self.protein_dict = protein_dict
        self.protein_interactions = []
        with open(protein_interactions_file, 'r') as f:
            for line in f:
                protein_a, protein_b, label = line.strip().split("\t")
                self.protein_interactions.append((protein_a, protein_b, int(label)))

    def __len__(self):
        return len(self.protein_interactions)

    def __getitem__(self, index):
        protein_a, protein_b, label = self.protein_interactions[index]
        return self.protein_dict[protein_a], self.protein_dict[protein_b], torch.tensor(label, dtype=torch.float)

def get_ppi_data_loader(dictionary, action_file, batch_size, shuffle):
    dataset = ProteinInteractionDataset(dictionary, action_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def ppi_train_step(model, data_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_samples = 0

    for proteinA, proteinB, labels in data_loader:
        optimizer.zero_grad()
        proteinA, proteinB, labels = proteinA.to(device), proteinB.to(device), labels.to(device)
        probabilities = model(proteinA, proteinB)
        loss = criterion(probabilities, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(proteinA)
        total_samples += batch_size

        total_loss += loss.item()*batch_size
    return total_loss / total_samples

def ppi_test_step(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for proteinA, proteinB, labels in data_loader:
            proteinA, proteinB, labels = proteinA.to(device), proteinB.to(device), labels.to(device)
            probabilities = model(proteinA, proteinB)
            loss = criterion(probabilities, labels)

            batch_size = len(proteinA)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            all_labels.append(labels)
            all_probabilities.append(probabilities)
        
    all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
    all_probabilities = torch.cat(all_probabilities, dim=0).detach().cpu().numpy()

    aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_classification_metrics(all_labels, all_probabilities)
    return total_loss / total_samples, aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc

def calculate_classification_metrics(label, probabilities):

    predictions = np.round(probabilities)

    aucroc = roc_auc_score(label, probabilities)
    true_positive_rate, false_positive_rate, _ = precision_recall_curve(label, probabilities)
    prc = auc(false_positive_rate, true_positive_rate)
    accuracy = accuracy_score(label, predictions)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(label, predictions).ravel()
    sensitivity = recall_score(label, predictions)
    specificity = true_negative / (true_negative + false_positive)
    precision = precision_score(label, predictions)
    f1 = f1_score(label, predictions)
    mcc = matthews_corrcoef(label, predictions)
    return aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc
