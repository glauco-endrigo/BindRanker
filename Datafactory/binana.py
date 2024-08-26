import time
from tqdm import tqdm
import os
from pathlib import Path
#from Config import Config
from BindRanker.Config import Config

config = Config()

pdbs = os.listdir(config.set) # ['1uz8']#
nposes = config.docking_params['nposes']
path_scripts = Path(os.getcwd())
binana_executable = config.executables['binana']

def binana(pdb, config):
    os.makedirs(config['output_folder'], exist_ok=True)  # Create the directory path if it doesn't exist
    binana_exec_cmd = f"python3 {binana_executable} -receptor {config['receptor_path']} -ligand {config['ligand_path']} -output_dir {config['output_folder']} > errors.txt"
    os.system(binana_exec_cmd)

receptor = "protein_fixed.pdb"
#receptor = 'protein.pdb'
start_time = time.time()  # Record start time

# Combined loop for both versions
for pdb in tqdm(pdbs, desc="Processing PDBS"):

    config_version_2 = {
        'ligand_path': f"{config.set}/{pdb}/ligand.pdb",
        'output_folder': f"{config.set}/{pdb}/binana/binana_{pdb}",
        'receptor_path': f"{config.set}/{pdb}/{pdb}_{receptor}"
    }

    for pose in range(1, nposes + 1):
        print('pose: ', pose)
        # Configuration dictionaries for each version
        config_version_1 = {
            'ligand_path': f"{config.set}/{pdb}/results/pose_{pose}.pdb",
            'output_folder': f"{config.set}/{pdb}/binana/binana_{pdb}_pose_{pose}",
            'receptor_path': f"{config.set}/{pdb}/{pdb}_{receptor}"
        }
        try:
            binana(pdb, config=config_version_1)
        except ValueError as e:
            print('Error: ', e)
            pass
    binana(pdb, config=config_version_2)

end_time = time.time()  # Record end time
duration_seconds = end_time - start_time  # Calculate duration in seconds
duration_hours = duration_seconds / 3600  # Convert duration to hours

# Save duration to a text file
with open(f"{config.data}/execution_time_binana.txt", "w") as file:
    file.write(f"Total execution time: {duration_hours:.2f} hours")
