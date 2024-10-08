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

        self.set = f"{self.root}/coreset_100"
        self.data = f"{self.root}/{self.project_name}/Data"

        if not os.path.exists(f"{self.data}"):
            os.makedirs(f"{self.data}")
        #self.pdb_list = pd.read_csv(f"../Datahub/pdbs_refined_set.csv")['PDB code'].to_list()[1:500]
        self.pdb_list = os.listdir('../coreset_100')#[:3]# ["4bps"] #for test
        self.model_args = {'seed': 42,
                           'savepath': f'{self.root}/{self.project_name}/',
                           'batch_size': 250,  #250
                           'epochs': 500 ,#300*2,
                           'nfolds': 3,
                           'patience': 15,
                           "lr": 0.05, #0.05,  # 0.05,
                           "gmp": 10,
                           'pos_weight': 13.5}
        # 250*10 Valor usado para refiened set
        self.label_args = {'class_def': "rmsd",
                           }
        self.node_descriptors = ["Mass"
            , "Charge"
            , "Element_Name"
            , "Hybridization"
            , "Formal_Charge"
            , "Unpaired_Electron"
            , "In_Aromatic_Substructure"]

        self.docking_params = {'exhaustiveness': 50,
                               'buffer_size': 10,
                               'nposes': 100}
        self.softwares = "../executables"
        self.executables = {'vina': f"{self.softwares}/vina_1.2.5_linux_x86_64",
                            "binana": f"{self.softwares}/binana-2.1/python/run_binana.py"}