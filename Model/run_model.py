from tqdm import tqdm
import pickle
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import subprocess
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from torch_geometric.loader import DataLoader
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import Data, Dataset
import os
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from BindRanker.Config import Config
from torch_geometric.nn import GATConv, global_mean_pool
config = Config()

patience = config.model_args["patience"]


####  GATModel
max_size = 650
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(GATModel, self).__init__()
        self.batch_size = batch_size
        self.conv1 = GATConv(input_dim, hidden_dim, heads=1)  # You can adjust the number of heads

        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)  # You can adjust the number of heads
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1)  # You can adjust the number of heads

        self.conv4 = GATConv(hidden_dim, self.batch_size, heads=1)  # Output dimension set to 1


        self.fc = nn.Linear(max_size, 1)  # Initialize fc layer as None

    def forward(self, data):
        x_s, x_t, edge_index, distances, batch = data.x_s, data.x_t, data.edge_index, data.edge_attr, data.batch

        # print("data.batch ALTERED: ", data.batch)
        #print(20 * "*")

        ## print("batch: ", data.batch.shape,"\n", data.batch)
        #print("x_s.shape: ", x_s.shape, "x_t.shape: ", x_t.shape, end='\n\n')
        ##print("x_s: ", x_s, "x_t: ", x_t, end='\n\n')

        # Pad the sequences to have the same length

        x = torch.cat((x_s, x_t), dim=0)  # Concatenate features
        #print(x[0:5], end='\n\n')
        #print("x.shape after cat", x.shape, end='\n\n')
        #print(20 * "*")

        target_size = max_size
        padding_needed = max(target_size - x.shape[0], 0)
        padding = (0, 0, 0, padding_needed)  # Assuming you want to pad the first dimension only
        x = F.pad(x, padding, mode='constant', value=0)
        #print("x.shape after PADDING", x.shape)
        #print("x after PADDING", x[0:5])
        #print(40 * "*")

        x = self.conv1(x, edge_index, edge_attr=distances)
        #print("x after conv1: ", x[0:5], end='\n\n')
        #print("x shape after conv1: ", x.shape, end='\n\n')
        x = torch.relu(x)
        #print("x.shape after relu 1", x.shape, end='\n\n')
        #print(40 * "*")

     #   x = self.conv2(x, edge_index, edge_attr=distances)
        #print("x shape after conv2: ", x.shape, end='\n\n')
     #   x = torch.relu(x)
        #print("x.shape after relu 2", x.shape, end='\n\n')
        #print(40 * "*")

       #  x = self.conv3(x, edge_index, edge_attr=distances)
        #print("x shape after conv 3: ", x.shape, end='\n\n')
       #  x = torch.relu(x)
        #print("x.shape after relu 3", x.shape, end='\n\n')
        #print(40 * "*")

        x = self.conv4(x, edge_index, edge_attr=distances)
        #print("x shape after conv 4: ", x.shape, end='\n\n')
        ## print("x after conv2: ", x, end='\n\n')
        x = F.leaky_relu(x, negative_slope=0.01)
        ##print("x.shape after relu 4", x.shape, end='\n\n')
        #print(40 * "*")

        x = x.view(self.batch_size, x.shape[0])
        #print("x.shape .view():", x.shape)
        ##print("x after .view():", x)
        #print("x shape after self.fc(x): ", self.fc(x).shape)
        x = self.fc(x)
        #print("x.shape after fc(x)", x.shape, end='\n\n')
        x = x.squeeze().unsqueeze(0)
        #print("x.shape after squeeze()", x.shape, end='\n\n')
        x = torch.sigmoid(x)
        ##print("output, after sigmoid", x, end='\n\n')

        #print(70 * '%*%')
        return x

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.fc]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


forward_desc = '''
    def forward(self, data):
        x_s, x_t, edge_index, distances, batch = data.x_s, data.x_t, data.edge_index, data.edge_attr, data.batch

        # print("data.batch ALTERED: ", data.batch)
        #print(20 * "*")

        ## print("batch: ", data.batch.shape,"\n", data.batch)
        #print("x_s.shape: ", x_s.shape, "x_t.shape: ", x_t.shape, end='\n\n')
        ##print("x_s: ", x_s, "x_t: ", x_t, end='\n\n')

        # Pad the sequences to have the same length

        x = torch.cat((x_s, x_t), dim=0)  # Concatenate features
        #print(x[0:5], end='\n\n')
        #print("x.shape after cat", x.shape, end='\n\n')
        #print(20 * "*")

        target_size = max_size
        padding_needed = max(target_size - x.shape[0], 0)
        padding = (0, 0, 0, padding_needed)  # Assuming you want to pad the first dimension only
        x = F.pad(x, padding, mode='constant', value=0)
        #print("x.shape after PADDING", x.shape)
        #print("x after PADDING", x[0:5])
        #print(40 * "*")

        x = self.conv1(x, edge_index, edge_attr=distances)
        #print("x after conv1: ", x[0:5], end='\n\n')
        #print("x shape after conv1: ", x.shape, end='\n\n')
        x = torch.relu(x)
        #print("x.shape after relu 1", x.shape, end='\n\n')
        #print(40 * "*")

     #   x = self.conv2(x, edge_index, edge_attr=distances)
        #print("x shape after conv2: ", x.shape, end='\n\n')
     #   x = torch.relu(x)
        #print("x.shape after relu 2", x.shape, end='\n\n')
        #print(40 * "*")

       #  x = self.conv3(x, edge_index, edge_attr=distances)
        #print("x shape after conv 3: ", x.shape, end='\n\n')
       #  x = torch.relu(x)
        #print("x.shape after relu 3", x.shape, end='\n\n')
        #print(40 * "*")

        x = self.conv4(x, edge_index, edge_attr=distances)
        #print("x shape after conv 4: ", x.shape, end='\n\n')
        ## print("x after conv2: ", x, end='\n\n')
        x = F.leaky_relu(x, negative_slope=0.01)
        ##print("x.shape after relu 4", x.shape, end='\n\n')
        #print(40 * "*")

        x = x.view(self.batch_size, x.shape[0])
        #print("x.shape .view():", x.shape)
        ##print("x after .view():", x)
        #print("x shape after self.fc(x): ", self.fc(x).shape)
        x = self.fc(x)
        #print("x.shape after fc(x)", x.shape, end='\n\n')
        x = x.squeeze().unsqueeze(0)
        #print("x.shape after squeeze()", x.shape, end='\n\n')
        x = torch.sigmoid(x)
        ##print("output, after sigmoid", x, end='\n\n')

        #print(70 * '%*%')
        return x
'''


# The heads parameter controls the number of attention heads used in the multi-head attention mechanism. Each attention head learns different relationships between nodes in the graph. When you set heads to 1, it means you are using only a single attention head for both convolution layers (self.conv1 and self.conv2).


#### Funtions to train and validate
# Define a function for the training loop with early stopping
def train_model_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, patience):
    model.train()
    total_loss = 0

    for batch_data in train_loader:  # This loop was missing
        optimizer.zero_grad()
        output = model(batch_data)
        target = batch_data.y
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)


# Define a function for the validation loop
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
            #print("batch_data: ", batch_data)
            #print("output: ", output)
            #print("target: ", target, end="\n\n")
            loss = criterion(output, target)
            val_loss += loss.item()
            val_true.extend(target.tolist())
            val_pred.extend((output > 0.5).float().tolist())  # Assuming binary classification
            val_probs.extend(output.tolist())

    accuracy = accuracy_score(val_true, val_pred)
    precision = precision_score(val_true, val_pred)
    recall = recall_score(val_true, val_pred)
    f1 = f1_score(val_true, val_pred)

    # Calculate AUC-ROC and AUC-PR
    auc_roc = roc_auc_score(val_true, val_probs)
    auc_pr = average_precision_score(val_true, val_probs)
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(val_true, val_pred)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(val_true, val_pred)

    # Calculate negative precision and recall
    TN = conf_matrix[0, 0]  # True negatives
    FN = conf_matrix[1, 0]  # False negatives
    TP = conf_matrix[1, 1]  # True positives
    FP = conf_matrix[0, 1]  # False positive
    neg_precision = TN / (TN + FN)
    neg_recall = TN / (TN + FP)  # Assuming FP is false positives

    TN, FN, TP, FP

    return val_loss / len(
        val_loader.dataset), accuracy, precision, recall, f1, balanced_acc, auc_roc, neg_precision, neg_recall, TN, FN, TP, FP


#### Compute fold metrics for each epoch
# Define a function to compute metrics for a given fold
def compute_fold_metrics(model, train_data, val_data, optimizer, criterion, num_epochs, batch_size):
    best_val_loss = float('inf')
    no_improvement = 0  # Counter for epochs with no improvement

    columns = ['Epoch', 'Validation Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', "balanced_acc", "auc_roc",
               "neg_precision", "neg_recall", "TN", "FN", "TP", "FP"]
    df_fold_metrics = pd.DataFrame(columns=columns)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        #print("epoch: ", epoch)
        # Training loop
        train_loss = train_model_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, patience)

        # Validation loop
        val_loss, accuracy, precision, recall, f1, balanced_acc, auc_roc, neg_precision, neg_recall, TN, FN, TP, FP = validate_model(
            model, val_loader, criterion)
        print('precision:', precision, 'recall:', recall, "TN:", TN, "FN:", FN, 'TP:', TP, 'FP:', FP)
        # print("val_loss: ", val_loss)
        # print("best_val_loss: ", best_val_loss)
        # Validation loop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement.")
            break

        df_fold_metrics = df_fold_metrics.append({'Epoch': epoch + 1
                                                     , 'Validation Loss': val_loss
                                                     , 'Accuracy': accuracy
                                                     , 'Precision': precision
                                                     , 'Recall': recall
                                                     , 'F1-Score': f1
                                                     , "balanced_acc": balanced_acc
                                                     , "auc_roc": auc_roc
                                                     , "neg_precision": neg_precision
                                                     , "neg_recall": neg_recall
                                                     , "TN": TN
                                                     , "FN": FN
                                                     , "TP": TP
                                                     , "FP": FP

                                                  }, ignore_index=True)

    return df_fold_metrics


#### Define a function to perform k-fold cross-validation
# Define a function to perform k-fold cross-validation
def k_fold_cross_validation(model, dataset_list, num_folds, batch_size, num_epochs):
    columns = ['Fold', 'Epoch', 'Validation Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score', "balanced_acc",
               "auc_roc", "neg_precision", "neg_recall"]
    df_metrics = pd.DataFrame(columns=columns)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset_list, [int(data.y) for data in dataset_list])):
        print(f"Fold {fold + 1}/{num_folds}")
        train_data = [dataset_list[i] for i in train_idx]
        val_data = [dataset_list[i] for i in val_idx]

        model.reset_parameters()
        fold_metrics = compute_fold_metrics(model, train_data, val_data, optimizer, criterion, num_epochs, batch_size)
        fold_metrics['Fold'] = fold + 1
        df_metrics = df_metrics.append(fold_metrics, ignore_index=True)
        #print(fold_metrics)
    return df_metrics


# %run Dataforge.ipynb

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, y=None, edge_attr=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.y = y
        self.edge_attr = edge_attr  # Add edge_attr attribute
        # self.num_nodes = x_s.size(0) +  x_t.size(0)
        self.num_nodes = (x_s.size(0) if x_s is not None else 0) + (x_t.size(0) if x_t is not None else 0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


# To load the data back with the correct data types
with open(f'{config.data}/bipartite_data.pkl', 'rb') as file:
    dataset_list = pickle.load(file)

##### Filter data list
filtered_data_list_num_nodes = [data for data in dataset_list if data.num_nodes > 0]
##### Filter data list
filtered_data_list_descriptors = [data for data in filtered_data_list_num_nodes if
                                  data.x_s.shape[0] > 0 and data.x_t.shape[0] > 0]
filtered_data_list = filtered_data_list_descriptors[0:2000]

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


# Initialize the model
model = GATModel(input_dim=7, hidden_dim=400, batch_size=config.model_args["batch_size"])

# Define loss and optimizer
#criterion = nn.BCEWithLogitsLoss()
criterion = BalancedBCEWithLogitsLoss(pos_weight=torch.tensor(12))
#optimizer = optim.Adam(model.parameters(), lr=config.model_args['lr'])
import time


import torch.optim as optim

# Assuming you have a model and a configuration named 'model' and 'config' respectively
optimizer = optim.SGD(model.parameters(), lr=config.model_args['lr'])

# Rest of your code using the optimizer...


# Main code
if __name__ == "__main__":
    # Define your model, optimizer, criterion, dataset_list, num_folds, batch_size, and num_epochs here
    start_time = time.time()
    # Perform k-fold cross-validation and get metrics
    df_metrics = k_fold_cross_validation(model, filtered_data_list, config.model_args["nfolds"],
                                         config.model_args["batch_size"], config.model_args["epochs"])

    # Calculate the mean metrics for each epoch across all folds
    mean_metrics = df_metrics.groupby(['Fold', 'Epoch']).mean().reset_index()

    # You can now work with 'df_metrics' for detailed metrics and 'mean_metrics' for mean metrics
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time

#print(mean_metrics.head(1))

## Result analysis
##### update_model_info Function

def get_and_increment_counter(counter_file):
    if os.path.exists(counter_file):
        with open(counter_file, "r") as file:
            counter = int(file.read())
    else:
        counter = 1

    with open(counter_file, "w") as file:
        file.write(str(counter + 1))

    return counter

def update_model_info(df, config):
    counter_file = "counter_file.txt"
    counter = get_and_increment_counter(counter_file)

    # Generate a unique model name based on the counter
    model_name = f"model_{counter}"

    # Assign dictionary values to DataFrame columns
    df['Model Args'] = [config.model_args] * len(df)
    df['Docking Params'] = [config.docking_params] * len(df)
    df['class_def'] = [config.label_args['class_def']] * len(df)
    df['batch'] = [config.model_args['batch_size']] * len(df)
    df["gmp"] = [config.model_args["gmp"]] * len(df)
    df['Model name'] = model_name
    df['forward_desc'] = [forward_desc] * len(df)  # You need to define forward_desc
    df['label_distribution'] = [label_distribution] * len(df)  # You need to define label_distribution
    df['amount_of_graphs_used_to_train'] = [amount_of_graphs_used_to_train] * len(df)  # You need to define amount_of_graphs_used_to_train
    df['node_descriptors'] = [config.node_descriptors] * len(df)
    df['execution_time (abs H)'] = [execution_time/3600] * len(df)
    df['execution_time (sec)'] = [execution_time] * len(df)

    #df['description'] = " "
    #df['date'] = " "

    # Print the current counter
    print(counter)


##### update_metrics_file
def update_metrics_file(df, metrics_file):
    if not os.path.exists(metrics_file):
        df.to_csv(metrics_file, index=False)
    else:
        existing_df = pd.read_csv(metrics_file)
        existing_df = pd.concat([df, existing_df], axis=0, ignore_index=True)
        existing_df.to_csv(metrics_file, index=False)

if not os.path.exists("../results"):
    os.makedirs("../results")

###### Best metrics
datos = mean_metrics.tail(1)
update_model_info(datos, config)
update_metrics_file(datos, metrics_file="../results/metrics_file_model.csv")

###### All metrics
update_model_info(mean_metrics, config)
update_metrics_file(mean_metrics, metrics_file="../results/all_metrics_file.csv")
all_metrics_from_all_models = pd.read_csv("../results/all_metrics_file.csv")

def plot_loss_vs_epochs(df_metrics):
    # Group the data by 'Epoch' and calculate the mean validation loss for each epoch across all folds
    mean_loss_by_epoch_all_folds = df_metrics.groupby('Epoch')['Validation Loss'].mean()
    # Create a list of epochs (x-axis) for the mean line
    epochs_mean = mean_loss_by_epoch_all_folds.index
    # Create a list of mean validation losses (y-axis) for the mean line
    losses_mean = mean_loss_by_epoch_all_folds.values
    # Plot the graph for the mean validation loss across all folds
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_mean, losses_mean, label='Mean Loss (All Folds)', marker='o', linestyle='-')
    # Plot individual validation loss lines for each fold
    for fold in df_metrics['Fold'].unique():
        fold_df = df_metrics[df_metrics['Fold'] == fold]
        mean_loss_by_epoch_fold = fold_df.groupby('Epoch')['Validation Loss'].mean()
        epochs_fold = mean_loss_by_epoch_fold.index
        losses_fold = mean_loss_by_epoch_fold.values
        plt.plot(epochs_fold, losses_fold, label=f'Fold {fold} Loss', linestyle='--')
    plt.title('Validation Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Usage:
# Assuming you have executed k-fold cross-validation and have 'df_metrics' available
plot_loss_vs_epochs(df_metrics)



###### Graph all results

# Assuming your DataFrame is named 'df'
# You can replace 'df' with the actual name of your DataFrame

# Create a Seaborn lineplot to visualize the data
plt.figure(figsize=(12, 6))
sns.lineplot(data=all_metrics_from_all_models, x='Epoch', y='Validation Loss', hue='Model name', estimator=None)

# Customize the plot
plt.title('Validation Loss by Epoch by Model Name')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend(title='Model Name', loc='upper right')

# Show the plot
plt.show()
