

def validate_atom_features():
    pass

def validate_bond_features():
    pass

from rdkit import Chem
from chembl_structure_pipeline import standardizer

# smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
# inchi = Chem.MolToInchiKey(mol)


# c = pubchempy.Compound.from_cid(a)
# c.isomeric_smiles        

def standardize_smiles(smi):
    smimol = Chem.MolFromSmiles(smi)
    parent, flag = standardizer.get_parent_mol(smimol)
    newmol = standardizer.standardize_mol(parent)
    smiles = Chem.MolToSmiles(newmol)
    return smiles

# print(check_smiles('C/C=C\CO'))
# print(check_smiles('C/C=CCO'))
# print(check_smiles('CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C'))
# print(check_smiles('N[C@@H](C)C(=O)O'))
# print(check_smiles('C[C@H](N)C(=O)O'))
