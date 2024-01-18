import os
import torch
import pandas as pd

class Config():
    def __init__(self):
        self.root = '..'
        self.project_name = 'Datahub'
        if not os.path.exists(f'{self.root}/{self.project_name}'):
            os.makedirs(f'{self.root}/{self.project_name}')

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
        self.create_dataset = False
        self.processed_dir = f"{self.root}/{self.project_name}/processed_dir"

        if not os.path.exists(f"{self.root}/{self.project_name}/processed_dir"):
            os.makedirs(f"{self.root}/{self.project_name}/processed_dir")

        self.set = f"{self.root}/coreset"
        self.data = f"{self.root}/{self.project_name}/Data"

        if not os.path.exists(f"{self.data}"):
            os.makedirs(f"{self.data}")
        self.pdb_list = list(os.listdir(self.set))

        self.model_args = {'seed': 42,
                           'savepath': f'{self.root}/{self.project_name}/',
                           'batch_size': 1,
                           'epochs': 20,
                           'nfolds': 5,
                           'patience': 15,
                           "lr": 0.01,
                           "gmp": 10,
                           }
        self.label_args = {'class_def': "rmsd",
                           }
        self.node_descriptors = ["Mass"
            , "Charge"
            , "Element_Name"
            , "Hybridization"
            , "Num_Hydrogens"
            , "Formal_Charge"
            , "Unpaired_Electron"
            , "In_Aromatic_Substructure"]

        self.docking_params = {'exhaustiveness': 50,
                               'buffer_size': 10,
                               'nposes': 10}
        self.softwares = "../executables"
        self.executables = {'vina': f"{self.softwares}/vina_1.2.5_linux_x86_64"
                            }