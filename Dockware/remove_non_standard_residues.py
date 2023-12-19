from Bio import PDB
from Config import Config
from tqdm import tqdm


config = Config()
PATH_mglt = config.softwares
pdb_list = config.pdb_list
print(pdb_list)

def remove_non_standard_residues(pdb_code):
    # Specify the input and output PDB files
    file = f'{config.coreset}/{pdb_code}/{pdb_code}_protein.pdb'

    # Create a structure object from the input PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file)

    # Iterate through the structure and remove non-standard residues
    for model in structure:
        for chain in model:
            residues_to_remove = [residue for residue in chain if not PDB.is_aa(residue)]
            for residue in residues_to_remove:
                chain.detach_child(residue.id)

    # Save the modified structure to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(file)

    print(f"Non-standard residues removed. Check the output file: {file}")


def main():
    for pdb_code in tqdm(pdb_list):
        remove_non_standard_residues(pdb_code)



if __name__ == "__main__":
    main()