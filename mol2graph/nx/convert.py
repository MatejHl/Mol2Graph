import networkx as nx
from rdkit import Chem

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


def fasta_to_nx(fasta, validate=False):
    """
    Convert FASTA string to nx.Graph.

    Notes:
    ------
    Internally this function creates RDkit molecule and uses mol_to_nx.
    """
    mol = Chem.MolFromFASTA(fasta.strip())
    can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
    G = mol_to_nx(mol)
    if validate:
        mol = nx_to_mol(G)
        new_smi = Chem.MolToSmiles(mol)
        assert new_smi == smiles
    return G


def pdb_to_nx():
    import os
    import io
    
    # biopython
    from Bio.PDB import PDBParser, PPBuilder, Chain
    
    
    raise NotImplementedError('Needs to be finished')
    
    
    def get_structure_by_path(pdb_id, pdb_path):
        parser = PDBParser(QUIET=False)
        return parser.get_structure(pdb_id, pdb_path)