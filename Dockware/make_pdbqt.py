
from BindRanker.Config import Config
import os
import shutil

config = Config()
PATH_mglt = config.softwares
pdb_codes_list =  config.pdb_list


def prepare_receptor(pdb_file):
    print("pdb_file:", pdb_file)
    receptor_name = pdb_file.split("/")[3]
    print("receptor_name:", receptor_name)
    output_file = f"receptor.pdbqt"
    print("output_file:", output_file)
    cmd = f"{PATH_mglt}/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {pdb_file} -o {output_file} -A hydrogen"
    print(cmd)
    os.system(cmd)
    return output_file


def process_pdb_receptor(pdb_code):
    pdb_file = f"{config.set}/{pdb_code}/{pdb_code}_protein.pdb"
    if os.path.isfile(pdb_file):
        treated_file = prepare_receptor(pdb_file)
        destination_file = os.path.join(os.path.dirname(pdb_file), os.path.basename(treated_file))
        shutil.move(treated_file, destination_file)
    else:
        print(f"File {pdb_file} not found!")


def prepare_ligand(pdb_file):
    receptor_name = os.path.splitext(pdb_file)[0]
    output_file = f"ligand.pdbqt"
    #cmd = f"{PATH_mglt}/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l {pdb_file} -o {output_file}"
    cmd = f"/home/lbcb02/Workspace/Scripts/executables/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l {pdb_file} -o {output_file}"
    os.system(cmd)
    return output_file


def process_pdb_ligand(pdb_code):
    folder_path = f"{config.set}/{pdb_code}"

    pdb_file = f"{pdb_code}_ligand.mol2"
    original_directory = os.getcwd()
    os.chdir(folder_path)
    if os.path.isfile(pdb_file):
        try:
            treated_file = prepare_ligand(pdb_file)
            destination_file = os.path.join(os.path.dirname(pdb_file), os.path.basename(treated_file))
            shutil.move(treated_file, destination_file)
            print("treated_file: ", treated_file)
            print("destination_file: ", destination_file)
        finally:
            # Change back to the original working directory even if an exception occurs
            os.chdir(original_directory)
    else:
        print(f"File {pdb_file} not found!")


def main():
    for pdb_code in pdb_codes_list:
        print("PDB", pdb_code)
        process_pdb_receptor(pdb_code)
        process_pdb_ligand(pdb_code)


if __name__ == "__main__":
    main()
