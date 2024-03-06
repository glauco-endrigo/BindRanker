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
        #self.pdb_list = pd.read_csv(f"../Datahub/pdbs_refined_set.csv")['PDB code'].to_list()[1:500]
        self.pdb_list = os.listdir('../coreset')#[:3]# ["4bps"] #for test
        self.model_args = {'seed': 42,
                           'savepath': f'{self.root}/{self.project_name}/',
                           'batch_size': 6,
                           'epochs': 200,
                           'nfolds': 3,
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