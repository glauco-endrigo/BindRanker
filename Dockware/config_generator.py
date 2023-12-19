from tqdm import tqdm
from pymol import cmd
import os
from Config import Config

config = Config()

def load_ligand(pdb_code):
    path_ligand = f"coreset/{pdb_code}/{pdb_code}_ligand.mol2"
    ligand_name = f"ligand_{pdb_code}"
    cmd.load(path_ligand, ligand_name)
    cmd.select("selection_name", ligand_name)
    output = cmd.centerofmass("selection_name")
    return output


def create_config_file(pdb_code, exhaustiveness, receptor_file="receptor.pdbqt", ligand_file="ligand.pdbqt",
                       size_x="30.0", size_y="30.0", size_z="30.0"):
    output = load_ligand(pdb_code)
    config_str = f"""receptor = {receptor_file}
ligand = {ligand_file}
center_x = {round(output[0], 2)}
center_y = {round(output[1], 2)}
center_z = {round(output[2], 2)}
exhaustiveness = {exhaustiveness}
"""
    coreset_folder = "coreset"
    pdb_folder = os.path.join(coreset_folder, pdb_code)
    os.makedirs(pdb_folder, exist_ok=True)
    config_file_path = os.path.join(pdb_folder, "config.txt")
    with open(config_file_path, "w") as config_file:
        config_file.write(config_str)


def calculate_box_dimensions_cmd(molecule_path, buffer_size):
    cmd.load(molecule_path, "molecule")
    extent = cmd.get_extent("molecule")
    min_x, min_y, min_z = extent[0]
    max_x, max_y, max_z = extent[1]
    x_size = max_x - min_x + buffer_size
    y_size = max_y - min_y + buffer_size
    z_size = max_z - min_z + buffer_size
    cmd.delete("molecule")
    return x_size, y_size, z_size


for pdb in tqdm(os.listdir("coreset"), desc="Processing PDBs"):
    print(pdb)
    create_config_file(pdb, config.docking_params['exhaustiveness'])
    subdir = os.path.join(os.getcwd(), "coreset", pdb)
    ligand_file_path = os.path.join(subdir, f"{pdb}_ligand.sdf")

    if os.path.exists(ligand_file_path):
        x_size, y_size, z_size = calculate_box_dimensions_cmd(ligand_file_path, config.docking_params['buffer_size'])
        config_file_path = os.path.join(subdir, "config.txt")
        if os.path.exists(config_file_path):
            config_str = f"""size_x = {round(x_size, 2)}
size_y = {round(y_size, 2)}
size_z = {round(z_size, 2)}
"""
            with open(config_file_path, "a") as config_file:
                config_file.write(config_str + "\n")
