import os
import csv
from tqdm import tqdm
from Bio import PDB
import numpy as np
from BindRanker.Config import Config

config = Config()
nposes = config.docking_params['nposes']
files_directory = config.coreset
data_directory = config.data
csv_file_path = os.path.join(data_directory, "rmsd.csv")

# Create the "Data" directory if it doesn't exist
os.makedirs(data_directory, exist_ok=True)

csv_exists = os.path.exists(csv_file_path)
print(csv_exists)

from rdkit import Chem
from rdkit.Chem import AllChem


def calculate_rmsd(docked_ligand_path, crystal_ligand_path):
    docked_ligand = Chem.MolFromPDBFile(docked_ligand_path)
    crystal_ligand = Chem.MolFromPDBFile(crystal_ligand_path)

    if docked_ligand is not None and crystal_ligand is not None:
        try:
            # Remove hydrogens
            docked_ligand = Chem.RemoveHs(docked_ligand)
            crystal_ligand = Chem.RemoveHs(crystal_ligand)

            # Calculate RMSD
            rmsd = AllChem.CalcRMS(docked_ligand, crystal_ligand)
            return rmsd
        except Exception as e:
            return f"Error calculating RMSD: {e}"
    else:
        return "Error loading ligand structures."


with open(csv_file_path, "a", newline="") as csv_file:
    fieldnames = ["pdb", "poserank", "RMSD"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    if not csv_exists:
        csv_writer.writeheader()

    for i, pdb in enumerate(tqdm(os.listdir(files_directory))):
        print(pdb)
        print(i)
        path_to_pdb = os.path.join(files_directory, pdb)
        path_to_results = os.path.join(path_to_pdb, "results")

        for pose_rank in range(1, nposes + 1):
            crystal_path = os.path.join(path_to_pdb, "ligand.pdb")
            docked_ligand_path = os.path.join(path_to_results, f"pose_{pose_rank}.pdb")

            #print("crystal: ", crystal_path)
            #print("docked_ligand: ", docked_ligand_path)
            try:
                rmsd = calculate_rmsd(docked_ligand_path, crystal_path)
                rmsd = round(rmsd, 2)
                csv_writer.writerow({"pdb": pdb, "poserank": pose_rank, "RMSD": str(rmsd)})
                #print(f'RMSD between pose and crystal (without hydrogens): {rmsd:.2f} Å')
            except Exception as e:
                print(f"Pose {pose_rank} does not exist:")