#from BindRanker.Config import Config
from Config import Config
from tqdm import tqdm
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem
from timeout_decorator import timeout, TimeoutError

config = Config()

PATH_mglt = config.softwares
pdb_list = config.pdb_list[100:]#


@timeout(30, use_signals=False)
def fix_pdb_structure(pdb_code):
    print('Trying to fix: ',pdb_code)
    # Specify the input and output PDB files
    input_file = f'{config.set}/{pdb_code}/{pdb_code}_protein.pdb'
    output_file = f'{config.set}/{pdb_code}/{pdb_code}_protein_fixed.pdb'
    try:
        Chem.MolFromPDBFile(input_file)
        print("protein.pdb was opened successfully")
    except Exception as e:
        print("Error when opening protein.pdb")
    # Create a PDBFixer instance
    pdbfixer = PDBFixer(filename=input_file)

    # Apply fixes
    pdbfixer.findMissingResidues()
    pdbfixer.findMissingAtoms()
    pdbfixer.addMissingAtoms()

    pdbfixer.addMissingHydrogens(7.0)  # Assuming a pH of 7.0

    # Convert non-standard residues to their standard equivalents
    pdbfixer.findNonstandardResidues()
    pdbfixer.replaceNonstandardResidues()

    # Delete unwanted heterogens
    pdbfixer.removeHeterogens(keepWater=False)  # Keeping water molecules

    # Save the corrected structure
    PDBFile.writeFile(pdbfixer.topology, pdbfixer.positions, open(output_file, 'w'))

    try:
        mol = Chem.MolFromPDBFile(output_file)
        print(f"{pdb_code}, Worked")

        if mol is not None:
            AllChem.RemoveHs(mol, implicitOnly=True)
            Chem.MolToPDBFile(mol, output_file)
    except Exception as e:
        print(f"Error when opening fixed file {pdb_code}")

    print(f"Structure issues fixed. Check the output file: {output_file}")

def main():
    erros = 0
    for pdb_code in tqdm(pdb_list):
        try:
            fix_pdb_structure(pdb_code)
        except TimeoutError:
            erros += 1
            print(f"Calculation for {pdb_code} timed out.")

            continue  # Skip to the next iteration
    print(f'Nº of errors {erros}')
if __name__ == "__main__":
    main()