import networkx as nx
from rdkit import Chem
import numpy as np
import scipy

# from mol2graph.containers import ExtendedGraph as sp_Graph
# from mol2graph.containers import ExtendedDataset

import mol2graph.nx.convert as convert_nx
import mol2graph.numpy.convert as convert_numpy
import mol2graph.spektral.convert as convert_spektral

print('\n WARNING: mol2graph.convert is DEPRICIATED and will be removed in the future. Use mol2graph.<framework>.convert instead.\n\n')

def mol_to_nx(mol):
    return convert_nx.mol_to_nx(mol)

def nx_to_mol(G):
    return convert_nx.nx_to_mol(G)

def smiles_to_nx(smiles, validate=False):
    return convert_nx.smiles_to_nx(smiles, validate = validate)

def fasta_to_nx(fasta, validate=False):
    return convert_nx.nx_to_mol(fasta, validate = validate)

def pdb_to_nx():
    return convert_nx.pdb_to_nx()
    

CHIRAL_TAG = {key : int(val) for key, val in Chem.rdchem.ChiralType.names.items()}
HYBRIDIZATION = {key : int(val) for key, val in Chem.rdchem.HybridizationType.names.items()}
BOND_TYPE = {key : int(val) for key, val in Chem.rdchem.BondType.names.items()}


def mol_to_numpy(mol, y = None, u = None, atom_features = [], bond_features = []):
    return convert_numpy.mol_to_numpy(mol, y = y, u = u, atom_features = atom_features, bond_features = bond_features)


def mol_to_spektral(mol, y = None, u = None, atom_features = [], bond_features = []):
    return convert_spektral.mol_to_spektral(mol = mol, 
                                            y = y, 
                                            u = u, 
                                            atom_features = atom_features, 
                                            bond_features = bond_features)


def spektral_to_mol():
    return convert_spektral.spektral_to_mol()


def smiles_to_spektral(smiles, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False,
                        atom_features = ['AtomicNum'], bond_features = ['BondType']):
    return convert_spektral.smiles_to_spektral(smiles = smiles, 
                                            y = y,
                                            u = u,
                                            validate = validate, 
                                            scipy_E = scipy_E, 
                                            IncludeHs = IncludeHs,
                                            atom_features = atom_features, 
                                            bond_features = bond_features)


def fasta_to_spektral(fasta, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False,
                    atom_features = ['AtomicNum'], bond_features = ['BondType']):
    return convert_spektral.fasta_to_spektral(fasta = fasta, 
                                            y = y, 
                                            u = u, 
                                            validate = validate, 
                                            scipy_E = scipy_E, 
                                            IncludeHs = IncludeHs,
                                            atom_features = atom_features, 
                                            bond_features = bond_features)
    
    
def smiles_to_numpy(smiles, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False,
                    atom_features = ['AtomicNum'], bond_features = ['BondType']):
    return convert_numpy.smiles_to_numpy(smiles = smiles,
                                        y = y, 
                                        u = u, 
                                        validate = validate, 
                                        scipy_E = scipy_E, 
                                        IncludeHs = IncludeHs,
                                        atom_features = atom_features, 
                                        bond_features = bond_features)



if __name__ == '__main__':
    # pdb_name = "Olfr263.B99991199.pdb"
    # pdb_path = os.path.join("pdb_test", pdb_name)

    # fn = get_structure_by_path("Olfr263", pdb_path)

    # smiles = 'C/C/1=C/CC/C(=C\[C@H]2[C@H](C2(C)C)CC1)/C'
    # smiles = 'F/C=C\F'
    smiles = 'C/C=C\CO'
    G = smiles_to_spektral(smiles, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False, 
                            atom_features = ['AtomicNum'], bond_features = ['BondType', 'Stereo'])
    print(G.x)
    print(G.a)
    print(G.e)
    print(G.u)
    print(G)
    print('--------------------------')

    # fasta = 'WHVSC'
    # G = fasta_to_spektral(fasta, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False)
    # print(G.x)
    # print(G.a)
    # print(G.e)
    # print(G.u)
    # print(G)
    # print('--------------------------')

    # x, a, e, y, u = smiles_to_numpy(smiles, y = 1, u = np.array([1,2,3]), validate=False, scipy_E = False, IncludeHs = False)
    # print(x)
    # print(a)
    # print(e)
    # print(y)
    # print(u)
    # print('--------------------------')