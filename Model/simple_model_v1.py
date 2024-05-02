from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from torch_geometric.data import Data, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
import pandas as pd
from BindRanker.Config import Config
from torch_geometric.nn import GATConv, global_mean_pool

config = Config()
patience = config.model_args["patience"]

from sklearn.metrics import precision_recall_curve, auc

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
        self.num_nodes = 12  # (x_s.size(0) if x_s is not None else 0) + (x_t.size(0) if x_t is not None else 0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

#bipartite_data tesded, worked
#bipartite_data_name_normalized: worked too!
#bipartite_data_with_pose_pred
#bipartite_data_with_pose_pred_sorted_by_family_encode
#bipartite_data_no_pose_rank_sorted_by_family
# bipartite_data_no_pose_rank_sorted_by_family_cut_3.5
#bipartite_data_no_pose_rank_sorted_by_family_cut_4_refined_set_encoded
#bipartite_data_no_pose_rank_NOT_sorted_by_family_cut_4_coreset_encoded_1
#bipartite_data_no_pose_rank_NOT_sorted_by_family_cut_4_coreset_1
#bipartite_data_shuffle
#bipartite_data_no_pose_rank_NOT_sorted_by_family_cut_4_coreset_1_RAND
#bipartite_data_no_pose_rank_cut_4_refined_set_encoded_RAND
# To load the data back with the correct data types

with open(f'{config.data}/bipartite_data_no_pose_rank_NOT_sorted_by_family_cut_4_coreset_1_RAND.pkl', 'rb') as file:

    dataset_list = pickle.load(file)

##### Filter data list
filtered_data_list_num_nodes = [data for data in dataset_list if data.num_nodes > 0]
##### Filter data list
filtered_data_list_descriptors = [data for data in filtered_data_list_num_nodes if
                                  data.x_s.shape[0] > 0 and data.x_t.shape[0] > 0]
filtered_data_list = filtered_data_list_descriptors#[0:1800]

#### Data info
label_distribution = dict(Counter([label.y.tolist() for label in filtered_data_list]))
amount_of_graphs_used_to_train = len(filtered_data_list)

## Dataset info

print(65 * "*")
# print("Nº Pdbs:", len(pdbs))
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


# Model definition
"""class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=1)
        self.conv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1)
        self.fc1 = nn.Linear(hidden_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, data):
        x_s, x_t, edge_index, distances, x_t_batch, x_s_batch = data.x_s, data.x_t, data.edge_index, data.edge_attr, data.x_t_batch, data.x_s_batch

        ##########################################################
        # Works
        x_new_t = self.conv1((x_s, x_t), edge_index, size=(x_s.size(0), x_t.size(0)), edge_attr=distances)
        x = torch.relu(x_new_t)
       ##########################################################
        #x_new_t = self.conv1((x_s, x_t), edge_index, size=(x_s.size(0), x_t.size(0)), edge_attr=distances)
        #x_new_t = torch.relu(x_new_t)
        #x_new_s = self.conv2((x_new_t, x_s), edge_index[torch.tensor([1, 0])], size=(x_new_t.size(0), x_s.size(0)),edge_attr=distances)
        #x = torch.relu(x_new_s)
        ##########################################################

        #x = F.leaky_relu(x_new_t, negative_slope=0.01)
        #print('x after second relu', x)

        x = global_mean_pool(x, x_s_batch)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.squeeze()

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
"""

"""
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=3)
        self.conv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1)
        self.fc1 = nn.Linear(12*3, 10)
        #self.bn1 = nn.BatchNorm1d(10)
        #self.fc2 = nn.Linear(10, 10)
        #self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x_s, x_t, edge_index, distances, x_t_batch, x_s_batch = data.x_s, data.x_t, data.edge_index, data.edge_attr, data.x_t_batch, data.x_s_batch
        x_new_t = self.conv1((x_s, x_t), edge_index, size=(x_s.size(0), x_t.size(0)), edge_attr=distances)
        x = torch.relu(x_new_t)

        ##########3
        #x_new_s = self.conv2((x_t, x_s), edge_index[torch.tensor([1, 0])], size=(x_t.size(0), x_s.size(0)), edge_attr=distances)
        #x = torch.relu(x_new_s)
        #####################
        x = global_mean_pool(x, x_s_batch)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc3(x)
        #print('x: ', x)
        return x.squeeze()
#
"""
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels=input_dim, out_channels=100, heads=2)
        #self.conv2 = GATConv(in_channels=100, out_channels=hidden_dim, heads=1)
        self.fc1 = nn.Linear(100*2, 10)
        self.fc3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x_s, x_t, edge_index, distances, x_t_batch, x_s_batch = data.x_s, data.x_t, data.edge_index, data.edge_attr, data.x_t_batch, data.x_s_batch
        x_new_t = self.conv1((x_s, x_t), edge_index, size=(x_s.size(0), x_t.size(0)), edge_attr=distances)
        x = torch.relu(x_new_t)

        ##########3
        #x_new_s = self.conv2((x_t, x_s), edge_index[torch.tensor([1, 0])], size=(x_t.size(0), x_s.size(0)), edge_attr=distances)
        #x = torch.relu(x_new_s)
        #####################
        x = global_mean_pool(x, x_s_batch)
        x = self.fc1(x)
        #x = self.fc2(x)

        #x = self.dropout(x)
        x = self.fc3(x)
        #print('x: ', x)
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
    with torch.no_grad():
        for batch_data in val_loader:
            output = model(batch_data)
            target = batch_data.y
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


# Initialize the model
model = GATModel(input_dim=7, hidden_dim=7, batch_size=config.model_args["batch_size"])
criterion = BalancedBCEWithLogitsLoss(pos_weight=torch.tensor(11.5))
optimizer = optim.Adam(model.parameters(), lr=config.model_args['lr'], weight_decay=0.01)
train_loader = DataLoader(filtered_data_list[:1500], batch_size=config.model_args["batch_size"], shuffle=False, follow_batch=['x_s', 'x_t'])
test_loader = DataLoader(filtered_data_list[1500:1900], batch_size=config.model_args["batch_size"], shuffle=False, follow_batch=['x_s', 'x_t'])
#[22121:27650]
#[1500:1900]
empty_df = pd.DataFrame(columns=['Epoch', 'Validation Loss', 'Train Loss', 'precision', 'recall','auc_pr', 'f1' ])
empty_df.to_csv('../results/df_metrics.csv', index=False)
existing_df = pd.read_csv("../results/df_metrics.csv").reset_index(drop=True)

best_auc_pr = float('-inf')  # Inicializa a melhor pontuação de AUC-PR como negativa infinita
best_model_path = "../Model/best_model.pt"  # Caminho para salvar o melhor modelo

# Training loop
for epoch in range(1, config.model_args['epochs']):
    train_loss = train()
    train_loss = train_loss.item()
    val_loss, accuracy, precision, recall, f1, balanced_acc, auc_roc, neg_precision, neg_recall, TN, FN, TP, FP, auc_pr = validate_model(
        model, test_loader, criterion)
    print('epoch:', epoch, 'precision:', precision, 'recall:', recall, "TN:", TN, "FN:", FN, 'TP:', TP, 'FP:', FP,
              'auc_pr:', auc_pr, 'f1: ',f1, 'auc_roc:', auc_roc)

    existing_df = existing_df.append(
        {'Fold': "1"
            , 'Epoch': epoch + 1
            , 'Validation Loss': val_loss
            , 'Train Loss': train_loss
            , 'precision:': precision
            , 'recall:': recall
            , 'auc_pr:': auc_pr
            , 'f1: ': f1
         },
        ignore_index=True)

    existing_df.to_csv("../results/df_metrics.csv", index=False)

    if auc_pr > best_auc_pr:
        torch.save(model.state_dict(), best_model_path)
        best_auc_pr = auc_pr
        print("Best model saved with AUC-PR:", best_auc_pr)