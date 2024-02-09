
from BindRanker.Config import Config
import os
import shutil
import pandas as pd
config = Config()
PATH_mglt = config.softwares + "/MGLTools-1.5.6"
#PATH_mglt_2 = "/usr/local/MGLTools-1.5.7"
pdb_codes_list = config.pdb_list


def prepare_receptor(pdb_file):
    print("pdb_file:", pdb_file)
    receptor_name = pdb_file.split("/")[3]
    print("receptor_name:", receptor_name)
    output_file = f"receptor.pdbqt"
    print("output_file:", output_file)
    cmd = f"{PATH_mglt}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {pdb_file} -o {output_file} -A hydrogen"
    print(cmd)
    os.system(cmd)
    return output_file


def process_pdb_receptor(pdb_code):
    pdb_file = f"{config.set}/{pdb_code}/{pdb_code}_protein_fixed.pdb"
    if os.path.isfile(pdb_file):
        treated_file = prepare_receptor(pdb_file)
        destination_file = os.path.join(os.path.dirname(pdb_file), os.path.basename(treated_file))
        shutil.move(treated_file, destination_file)
    else:
        print(f"File {pdb_file} not found!")


def prepare_ligand(pdb_file):
    receptor_name = os.path.splitext(pdb_file)[0]
    output_file = f"ligand.pdbqt"
    cmd = f"{PATH_mglt}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l {pdb_file} -o {output_file}"
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


def save_problem_pdb(pdb_to_extend, file_name):
    try:
        if not os.path.exists(file_name):
            pd.DataFrame({'PDB Codes': pdb_to_extend}).to_csv(file_name, index=False)
            print(f"New file '{file_name}' created successfully.")
        else:
            existing_df = pd.read_csv(file_name)
            existing_list = existing_df['PDB Codes'].tolist()
            extended = existing_list + pdb_to_extend
            updated_df = pd.DataFrame({'PDB Codes': extended})
            updated_df.to_csv(file_name, index=False)
            print(f"File '{file_name}' extended successfully.")
    except Exception as e:
        print(f"An error occurred while saving '{file_name}': {e}")

def main():
    for pdb_code in pdb_codes_list:
        print("PDB", pdb_code)
        try:
            process_pdb_receptor(pdb_code)
        except Exception as e:
            print(f"Error processing PDB {pdb_code}: {e}")
            save_problem_pdb([pdb_code], 'pdb_error_when_make_pdbqt.csv')
            continue
        process_pdb_ligand(pdb_code)


if __name__ == "__main__":
    main()
