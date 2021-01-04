import itertools
import networkx as nx
from spektral.data.graph import Graph as sp_Graph
from rdkit import Chem
import numpy as np
import scipy


def mol_to_nx(mol):
    """
    For now, this is just an example function. Copy it and change if you need 
    any other Atom/Bond attributes.

    See 'rdkit.Chem.rdchem.Atom' for all possible attributes of nodes and
    'rdkit.Chem.rdchem.Bond' for all possible attributes of nodes 
    at https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

    Paramters:
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule

    Returns:
    --------
    graph : 
        Attributes of the graph are following:
        X = [atomic_num
             formal_charge
             chiral_tag
             hybridization
             num_explicit_hs
             is_aromatic]

        E = [bond_type]
        if scipy_E is True then E is scipy.sparse.coo.coo_matrix of shape
        (n_nodes, n_nodes, n_edge_features) which is symmetric in coordinates [0, 1]
        else it is a numpy.ndarray

        Categorical features in X and E are encoded by rdkit.Chem.rdchem standard numbering and
        mapping from numbers to names can be accessed from CHIRAL_TAG, HYBRIDIZATION and BOND_TYPE
        objects (dictionaries).
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol


def smiles_to_nx(smiles, validate=False):
    """
    Convert SMILES string to nx.Graph.

    Notes:
    ------
    Internally this function creates RDkit molecule and uses mol_to_nx.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
    G = mol_to_nx(mol)
    if validate:
        mol = nx_to_mol(G)
        new_smi = Chem.MolToSmiles(mol)
        assert new_smi == smiles
    return G


CHIRAL_TAG = {key : int(val) for key, val in Chem.rdchem.ChiralType.names.items()}
HYBRIDIZATION = {key : int(val) for key, val in Chem.rdchem.HybridizationType.names.items()}
BOND_TYPE = {key : int(val) for key, val in Chem.rdchem.BondType.names.items()}

def mol_to_spektral(mol, y = None):
    """
    For now, this is just an example function. Copy it and change if you need 
    any other Atom/Bond attributes.

    See 'rdkit.Chem.rdchem.Atom' for all possible attributes of nodes and
    'rdkit.Chem.rdchem.Bond' for all possible attributes of nodes 
    at https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

    Paramters:
    ----------
    mol : rdkit.Chem.rdchem.Mol

    Returns:
    --------
    graph : 
        Attributes of the graph are following:
        X = [atomic_num
             formal_charge
             chiral_tag
             hybridization
             num_explicit_hs
             is_aromatic]

        E = [bond_type]
        E is stored as a numpy array where elements are ordered by (row_idx, col_idx) (so similar 
        to scipy.sparse.crs_matrix) and adjecancy matrix is used to extraxt which to which edge the
        data corresponds. 

        Categorical features in X and E are encoded by rdkit.Chem.rdchem standard numbering and
        mapping from numbers to names can be accessed from CHIRAL_TAG, HYBRIDIZATION and BOND_TYPE
        objects (dictionaries).
    """
    A = Chem.GetAdjacencyMatrix(mol)
    X = []
    begin = []
    end = []
    E = []
    for atom in mol.GetAtoms():
        X.append([atom.GetAtomicNum(),          # atomic_num
                atom.GetFormalCharge(),         # formal_charge
                int(atom.GetChiralTag()),       # chiral_tag
                int(atom.GetHybridization()),   # hybridization 
                atom.GetNumExplicitHs(),        # num_explicit_hs
                int(atom.GetIsAromatic())])     # is_aromatic
    for bond in mol.GetBonds():
        begin.append(bond.GetBeginAtomIdx())
        end.append(bond.GetEndAtomIdx())
        E.append([int(bond.GetBondType())])

    _E = list(zip(begin + end, end + begin, E + E))
    _E.sort(key=lambda ele: (ele[0], ele[1]))
    E = np.array([dat[2] for dat in _E], dtype=np.float32)

    assert len(E.shape) == 2

    # data = list(itertools.chain.from_iterable(E))
    # E = scipy.sparse.csr_matrix((data + data, (begin + end, end + begin)), shape = Chem.GetAdjacencyMatrix(mol).shape)
    # E.sort_indices()

    G = sp_Graph(x = np.array(X, dtype = np.float32),
                a = A.astype(np.float32),
                e = E,
                y = y)
    return G


def spektral_to_mol():
    pass


def smiles_to_spektral(smiles, y = None, validate=False, scipy_E = False):
    """
    Convert SMILES string to nx.Graph.

    TO DO: implement validate

    Paramters:
    ----------
    validate : bool
        whether to validate if reverse conversion is working. This is not
        implemented at the moment.

    Notes:
    ------
    Internally this function creates RDkit molecule and uses mol_to_spektral(mol, scipy_E = False).
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    assert mol is not None

    can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
    G = mol_to_spektral(mol, y = y)
    if validate:
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")
    return G


def pdb_to_nx():
    import os
    import io
    
    # biopython
    from Bio.PDB import PDBParser, PPBuilder, Chain
    
    
    raise NotImplementedError('Needs to be finished and moved to smiles2graph')
    
    
    def get_structure_by_path(pdb_id, pdb_path):
        parser = PDBParser(QUIET=False)
        return parser.get_structure(pdb_id, pdb_path)
    
    if __name__ == '__main__':
        pdb_name = "Olfr263.B99991199.pdb"
        pdb_path = os.path.join("pdb_test", pdb_name)
    
        fn = get_structure_by_path("Olfr263", pdb_path)