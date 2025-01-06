import os.path
import pickle
import random
from collections import Counter
from datetime import date
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from BindRanker.Config import Config

config = Config()
patience = config.model_args["patience"]
print('Description of the experiment')
descricao = "Empty"# input()
from sklearn.metrics import precision_recall_curve, auc
def get_and_increment_counter(counter_file):
    if os.path.exists(counter_file):
        with open(counter_file, "r") as file:
            counter = int(file.read())
    else:
        counter = 1

    with open(counter_file, "w") as file:
        file.write(str(counter + 1))

    return counter

def calculate_percentage_distribution(counter, total):
    return {k: f"{(v / total * 100):.2f}%" for k, v in counter.items()}
##counter_file = "counter_file.txt"
##counter = get_and_increment_counter(counter_file)
##model_name = f"model_{counter}"

class BalancedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, input, target):
        loss = self.bce_loss(input, target)
        return loss


# %run Dataforge.ipynb

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, y=None, edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.y = y
        self.edge_attr = edge_attr  # Add edge_attr attribute
        # self.num_nodes = len(set(edge_index[0].tolist())) +  len(set(edge_index[1].tolist()))
        self.num_nodes = 9  # (x_s.size(0) if x_s is not None else 0) + (x_t.size(0) if x_t is not None else 0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

##############################################################################333
with open(f'{config.data}/coreset_26_11_2024.pkl', 'rb') as file:

    dataset_list = pickle.load(file)

print(Counter([data.y.tolist() for data in dataset_list]))

filtered_data_list_num_nodes = [data for data in dataset_list if data.num_nodes > 0]

filtered_data_list_descriptors = [data for data in filtered_data_list_num_nodes if
                                  data.x_s.shape[0] > 0 and data.x_t.shape[0] > 0]
filtered_data_list = filtered_data_list_descriptors#[0:1800]


#### Data info
label_distribution = dict(Counter([label.y.tolist() for label in filtered_data_list]))
amount_of_graphs_used_to_train = len(filtered_data_list)

## Dataset info
print(65 * "*")
#print("Nº Pdbs:", len(pdbs))
print("Nº BipartiteData objects:", len(dataset_list))
print("Nº BipartiteData objects filtered by num_nodes > 0: ", len(filtered_data_list_num_nodes))
print("Nº BipartiteData objects filtered by has descriptors > 0:", len(filtered_data_list_descriptors))
print("Nº BipartiteData objects for training", len(filtered_data_list))
print(65 * "*")


## Run Model

class BalancedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, input, target):
        loss = self.bce_loss(input, target)
        return loss



#### GATModel

# Get today's date
from torch_geometric.nn import global_mean_pool


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=9, heads=1)
        self.conv2 = GATConv(in_channels=9,out_channels=50, heads=1)
        self.fc1 = nn.Linear((50), 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x_s, x_t, edge_index, distances, x_t_batch, x_s_batch = data.x_s, data.x_t, data.edge_index, data.edge_attr, data.x_t_batch, data.x_s_batch
        x_new_t = self.conv1((x_s, x_t), edge_index, size=(x_s.size(0), x_t.size(0)), edge_attr=distances)
        x_new_s = self.conv2((x_new_t, x_s), edge_index, size=(x_new_t.size(0), x_s.size(0)), edge_attr=distances)
        x = torch.relu(x_new_s)
        x = global_mean_pool(x, x_s_batch)
        x = self.fc1(x)

        return x.squeeze()
# Training and validation functions
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach()

    return total_loss / len(train_loader.dataset)
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_true = []
    val_pred = []
    val_probs = []

    #print('Entering the loop')
    with torch.no_grad():
        for batch_data in val_loader:
            #print('batch_data:', batch_data)
            output = model(batch_data)
            target = batch_data.y
            #print('batch_data: ',batch_data)
            loss = criterion(output, target)
            val_loss += loss.item()
            val_true.extend(target.tolist())
            val_pred.extend((output > 0.5).float().tolist())
            val_probs.extend(output.tolist())
            del batch_data

    precision, recall, thresholds = precision_recall_curve(val_true, val_probs)
    auc_pr = auc(recall, precision)
    accuracy = accuracy_score(val_true, val_pred)
    precision = round(precision_score(val_true, val_pred), 2)
    recall = round(recall_score(val_true, val_pred), 2)
    f1 = round(f1_score(val_true, val_pred), 2)
    auc_roc = round(roc_auc_score(val_true, val_probs), 2)
    auc_pr = round(average_precision_score(val_true, val_probs), 2)
    balanced_acc = balanced_accuracy_score(val_true, val_pred)
    conf_matrix = confusion_matrix(val_true, val_pred)
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    neg_precision = TN / (TN + FN)
    neg_recall = TN / (TN + FP)
    return val_loss / len(
        val_loader.dataset), accuracy, precision, recall, f1, balanced_acc, auc_roc, neg_precision, neg_recall, TN, FN, TP, FP, auc_pr

for run in range(1, 6):  # Loop para rodar o modelo 5 vezes

    """indices = list(range(len(filtered_data_list)))
    random.shuffle(indices)
    filtered_data_list =  [filtered_data_list[i] for i in indices]"""

    print(f"Run {run}/5")
    counter_file = "counter_file.txt"
    counter = get_and_increment_counter(counter_file)
    model_name = f"model_{counter}"
    # Inicializa o modelo
    pos_weight = config.model_args["pos_weight"]
    model = GATModel(input_dim=9, hidden_dim=9, batch_size=config.model_args["batch_size"])
    criterion = BalancedBCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = optim.Adam(model.parameters(), lr=config.model_args['lr'], weight_decay=config.model_args['weight_decay'])

    """# Calcula os índices para divisão 70%/30%
    n = len(filtered_data_list)
    train_size = int(0.7 * n)
    print(filtered_data_list)
    # Define os loaders
    train_loader = DataLoader(
        filtered_data_list[:train_size],
        batch_size=config.model_args["batch_size"],
        shuffle=False,
        follow_batch=['x_s', 'x_t']
    )
    test_loader = DataLoader(
        filtered_data_list[train_size:],
        batch_size=config.model_args["batch_size"],
        shuffle=False,
        follow_batch=['x_s', 'x_t']
    )"""
    labels = [int(data.y.item()) for data in filtered_data_list]
    overall_counter = Counter(labels)
    overall_percentage = calculate_percentage_distribution(overall_counter, len(labels))

    # Define optional parameters
    split_kwargs = {
        "test_size": 0.3,  # Proportion for the test set
        "random_state": 42  # Ensure reproducibility
    }

    # Add stratify argument conditionally
    if config.model_args['stratify']:
        split_kwargs["stratify"] = labels  # Use class labels to stratify

    # Perform the split
    train_indices, test_indices = train_test_split(
        list(range(len(filtered_data_list))),  # Indices of the dataset
        **split_kwargs  # Pass parameters dynamically
    )

    train_data = [filtered_data_list[i] for i in train_indices]
    test_data = [filtered_data_list[i] for i in test_indices]

    train_loader = DataLoader(
        train_data,
        batch_size=config.model_args["batch_size"],
        shuffle=False,
        follow_batch=['x_s', 'x_t']
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.model_args["batch_size"],
        shuffle=False,
        follow_batch=['x_s', 'x_t']
    )
    if not os.path.exists('../results/df_metrics.csv'):
        empty_df = pd.DataFrame(columns=['Epoch', 'Validation Loss', 'Train Loss'])
        empty_df.to_csv('../results/df_metrics.csv', index=False)

    existing_df = pd.read_csv("../results/df_metrics.csv").reset_index(drop=True)

    best = float('-inf')  # Inicializa a melhor pontuação de AUC-PR como negativa infinita
    best_model_path = f"../Model/models_pt/{model_name}_run{run}.pt"  # Caminho para salvar o melhor modelo de cada run

    param = "f1"
    # Training loop
    for epoch in range(1, config.model_args['epochs']):
        train_loss = train()
        train_loss = train_loss.item()

        val_loss, accuracy, precision, recall, f1, balanced_acc, auc_roc, neg_precision, neg_recall, TN, FN, TP, FP, auc_pr = validate_model(
            model, test_loader, criterion)
        print('epoch:', epoch, 'precision:', precision, 'recall:', recall, "TN:", TN, "FN:", FN, 'TP:', TP, 'FP:', FP,
                  'auc_pr:', auc_pr, 'f1: ', f1, 'auc_roc:', auc_roc)

        existing_df = existing_df.append(
            {   'opt param' : param
                , 'lr': config.model_args['lr']
                , 'weight_decay': config.model_args['weight_decay']
                ,'model_name':model_name
                ,'Model Args':config.model_args
                ,'descricao':descricao
                ,'batch':config.model_args['batch_size']
                ,'Distribution':label_distribution
                ,'Qtd_graphs':amount_of_graphs_used_to_train
                ,'Fold': f"Run {run}"
                , 'Epoch': epoch + 1
                , 'Validation Loss': val_loss
                , 'Train Loss': train_loss
                , 'precision:': precision
                , 'recall:': recall
                , 'auc_pr:': auc_pr
                , 'f1: ': f1
                , 'auc_roc': auc_roc,
                "TN:": TN, "FN:": FN, 'TP:': TP, 'FP:': FP, "date": date.today()
                , 'indices':test_indices
                , "overall_percentage": overall_percentage
                , 'stratify':config.model_args['stratify']
                , "pos_weight":config.model_args['pos_weight']

             },
            ignore_index=True)

        metric = f1
        if metric > best:
            torch.save(model.state_dict(), best_model_path)
            best = metric
            print(f"Best model saved with {param}:", metric)

    df = existing_df.copy()

    df.to_csv("../results/df_metrics.csv", index=False)

    print(f"Completed Run {run}/5")

print("Training completed for all runs")

print(model_name)