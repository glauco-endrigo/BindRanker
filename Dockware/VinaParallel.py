import pandas as pd
import os
from Academy.Config import Config
import multiprocessing as mp
from tqdm import tqdm
import time
import logging
from datetime import datetime
config = Config()

nposes = config.docking_params['nposes']
PATH_vina = config.executables['vina']
pdb_codes_list = config.pdb_list

# Set up logging
logging.basicConfig(filename='docking_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_docking(args):
    folder, log_queue = args
    folder_path = os.path.join(config.coreset, folder)


    if os.path.isdir(folder_path):

        ligand_path = os.path.join(folder_path, "ligand.pdbqt")
        protein_path = os.path.join(folder_path, "receptor.pdbqt")
        config_file = os.path.join(folder_path, "config.txt")

        if os.path.exists(config_file):
            output_dir = os.path.join(folder_path, "results")
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.time()
            start_time_date = datetime.now()

            try:
                docking_command = f"{PATH_vina} --config {config_file} --ligand {ligand_path} --receptor {protein_path} --out {output_dir}/result.pdbqt  --num_modes 10"
                os.system(docking_command)

                end_time_date = datetime.now()
                result_file = os.path.join(output_dir, "result.pdbqt")

                elapsed_time = (end_time_date - start_time_date).total_seconds() / 60
                if os.path.exists(result_file):
                    logging.info(f"Docking successful for - {folder} - elapsed_time (min) - {elapsed_time}")
                else:
                    logging.error(f"Docking failed for {folder}! Result not generated.")



                log_entry = [folder
                    , start_time_date
                    , end_time_date
                    , elapsed_time
                    , config.docking_params['exhaustiveness']
                    , config.docking_params['buffer_size']]
                log_queue.append(log_entry)

            except Exception as e:
                logging.error(f"Error in docking for {folder_path}: {str(e)}")


if __name__ == '__main__':
    # Set up multiprocessing manager
    manager = mp.Manager()
    log_queue = manager.list()

    # Create a pool of processes
    pool = mp.Pool(processes=mp.cpu_count())
    #pool = mp.Pool(processes=2)
    # Use tqdm for the progress bar
    for _ in tqdm(pool.imap_unordered(perform_docking, [(code, log_queue) for code in os.listdir(config.coreset)]),
                   desc="Processing PDB codes"):
        pass

    # Close the pool
    pool.close()
    pool.join()

    # Save log entries to a CSV file
    log_columns = ['PDB Name', 'Start Time', 'End Time', 'Elapsed Time (min)', 'exhaustiveness', 'buffer_size']
    log_df = pd.DataFrame(list(log_queue), columns=log_columns)
    log_df.to_csv('docking_log.csv', mode='a', header=not os.path.exists('docking_log.csv'), index=False)
