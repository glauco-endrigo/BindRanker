import os
import subprocess
from tqdm import tqdm
from BindRanker.Config import Config
config = Config()

nposes = config.docking_params['nposes']
PATH_vina = config.executables['vina']
pdb_codes_list = config.pdb_list[0:5] #### Skpped number 55
PATH_coreset = config.coreset

print(os.listdir(PATH_coreset))
# Percorrendo as pastas dentro da pasta "coreset"
for folder in tqdm(os.listdir(PATH_coreset), desc="Processing PDB codes"):# os.listdir(coreset_path):
    folder_path = os.path.join(PATH_coreset, folder)

    # Verificando se é um diretório
    if os.path.isdir(folder_path):
        output_dir = os.path.join(folder_path, "results")
        result_path = os.path.join(output_dir, "result.pdbqt")
        # Lendo o arquivo de resultado e separando as poses
        with open(result_path, "r") as f: # File being read in read mode r
            content = f.read()
        poses = content.split("ENDMDL")
        # Salvando cada pose em um arquivo separado
        for idx, pose in enumerate(poses):
            if idx + 1 > nposes:
                break

            print(idx + 1)

            pose_filename = os.path.join(output_dir, f"pose_{idx + 1}.pdbqt")
            with open(pose_filename, "w") as f_pose:
                f_pose.write(pose.strip() + "ENDMDL\n")


print("Separação das poses concluída para todos os ligantes e proteínas!")


# Loop through folders in the coreset directory
for folder in tqdm(os.listdir(PATH_coreset), desc="Processing PDB codes"):  # Assuming you've defined pdb_codes_list somewhere
    folder_path = os.path.join(PATH_coreset, folder)
    print(folder)

    # print("---------------------"+folder+"---------------------")
    ligand_name = os.path.join(folder_path, f"ligand.pdbqt")
    # print(ligand_name)
    pdb_output_ligand = os.path.join(folder_path, f"ligand.pdb")
    # print(pdb_output_ligand)
    subprocess.run(["obabel", ligand_name, "-O", pdb_output_ligand])

    # Check if it's a directory
    if os.path.isdir(folder_path):
        output_dir = os.path.join(folder_path, "results")
        result_path = os.path.join(output_dir, "result.pdbqt")

        # Convert each pose to PDB format using obabel
        for idx in range(1, nposes + 1):
            pose_filename = os.path.join(output_dir, f"pose_{idx}.pdbqt")
            pdb_output_filename = os.path.join(output_dir, f"pose_{idx}.pdb")
            subprocess.run(["obabel", pose_filename, "-O", pdb_output_filename])

# print("Conversion from PDBQT to PDB completed for all ligands and proteins!")