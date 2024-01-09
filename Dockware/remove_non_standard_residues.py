import os

from Bio import PDB
from BindRanker.Config import Config
from tqdm import tqdm
from pdbfixer import PDBFixer
from openmm.app import PDBFile

config = Config()

PATH_mglt = config.softwares
pdb_list = config.pdb_list


def fix_pdb_structure(pdb_code):
    # Specify the input and output PDB files
    input_file = f'{config.coreset}/{pdb_code}/{pdb_code}_protein.pdb'
    output_file = f'{config.coreset}/{pdb_code}/{pdb_code}_protein_fixed.pdb'

    # Create a PDBFixer instance
    pdbfixer = PDBFixer(filename=input_file)

    # Apply fixes
    pdbfixer.findMissingResidues()
    pdbfixer.findMissingAtoms()
    pdbfixer.addMissingAtoms()

    # Add missing heavy atoms
    pdbfixer.addMissingHydrogens(7.0)  # Assuming a pH of 7.0

    # Convert non-standard residues to their standard equivalents
    pdbfixer.findNonstandardResidues()
    pdbfixer.replaceNonstandardResidues()

    # Delete unwanted heterogens
    pdbfixer.removeHeterogens(keepWater=True)  # Keeping water molecules

    # Save the corrected structure
    PDBFile.writeFile(pdbfixer.topology, pdbfixer.positions, open(output_file, 'w'))

    print(f"Structure issues fixed. Check the output file: {output_file}")


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
        # First, fix the structure
        fix_pdb_structure(pdb_code)
        #remove_non_standard_residues(pdb_code)


if __name__ == "__main__":
    main()