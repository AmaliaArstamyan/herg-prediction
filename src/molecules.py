import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold


def parse_smiles(smiles: str):
    """Parse a SMILES string, returning None when it is invalid."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def morgan_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    molecule = parse_smiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius, nBits=n_bits
    )
    result = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def scaffold_key(smiles: str) -> str:
    molecule = parse_smiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
    if scaffold_smiles:
        return scaffold_smiles
    return f"acyclic:{Chem.MolToSmiles(molecule, canonical=True)}"


def calculate_descriptors(smiles: str) -> dict[str, float | int]:
    """Calculate a small set of interpretable descriptors for a SMILES string."""
    molecule = parse_smiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    return {
        "Molecular weight (Da)": round(Descriptors.MolWt(molecule), 2),
        "cLogP": round(Descriptors.MolLogP(molecule), 2),
        "TPSA (A^2)": round(Descriptors.TPSA(molecule), 2),
        "Hydrogen-bond donors": int(Lipinski.NumHDonors(molecule)),
        "Hydrogen-bond acceptors": int(Lipinski.NumHAcceptors(molecule)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(molecule)),
    }
