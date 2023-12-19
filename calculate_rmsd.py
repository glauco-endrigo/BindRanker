import os
import csv
from tqdm import tqdm
from Bio import PDB
import numpy as np
from Config import Config

config = Config()
nposes = config.docking_params['nposes']
files_directory = "Run/coreset"

csv_file_path = "Data/rmsd.csv"
csv_exists = os.path.exists(csv_file_path)


def remove_hydrogens(structure):
    # Create a list to hold non-hydrogen atoms
    non_hydrogen_atoms = []

    for atom in structure.get_atoms():
        if atom.element != 'H':
            non_hydrogen_atoms.append(atom)

    return non_hydrogen_atoms


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
            print("pose:", pose_rank)
            crystal_path = os.path.join(path_to_pdb, "ligand.pdb")
            docked_ligand_path = os.path.join(path_to_results, f"pose_{pose_rank}.pdb")

            parser = PDB.PDBParser()

            print("crystal: ", crystal_path)
            print("docked_ligand: ", docked_ligand_path)

            try:
                pose_structure = parser.get_structure('pose', docked_ligand_path)
                crystal_structure = parser.get_structure('crystal', crystal_path)
            except Exception as e:
                print(e)
                continue

            # Remove hydrogen atoms from both structures
            pose_structure_atoms = remove_hydrogens(pose_structure)
            crystal_structure_atoms = remove_hydrogens(crystal_structure)

            # Make sure both structures have the same number of atoms for comparison.
            if len(pose_structure_atoms) != len(crystal_structure_atoms):
                print("The two structures have different numbers of non-hydrogen atoms.")
            else:
                # Calculate the RMSD between the non-hydrogen atoms.
                rmsd = np.sqrt(np.mean(np.square(np.array([a1.coord for a1 in pose_structure_atoms]) - np.array(
                    [a2.coord for a2 in crystal_structure_atoms]))))
                csv_writer.writerow({"pdb": pdb, "poserank": pose_rank, "RMSD": f"{rmsd:.2f}"})
                print(f'RMSD between pose and crystal (without hydrogens): {rmsd:.2f} Å')