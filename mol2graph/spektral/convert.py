from rdkit import Chem

from mol2graph.spektral.containers import ExtendedGraph as sp_Graph

from mol2graph.numpy.convert import mol_to_numpy

def mol_to_spektral(mol, y = None, u = None, atom_features = [], bond_features = []):
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
    x, a, e, y, u = mol_to_numpy(mol, y, u, atom_features, bond_features)
    G = sp_Graph(x = x,
                a = a,
                e = e,
                y = y,
                u = u)         
    return G


def spektral_to_mol():
    raise NotImplementedError('spektral_to_mol is not implemented yet.')


def smiles_to_spektral(smiles, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False,
                        atom_features = ['AtomicNum'], bond_features = ['BondType']):
    """
    Convert SMILES string to spektral.data.graph.Graph.

    TO DO: implement validate

    Paramters:
    ----------
    smiles : str
        SMILES string that will be converted to rdkit.Chem.rdchem.Mol using Chem.MolFromSmiles

    validate : bool
        whether to validate if reverse conversion is working. This is not
        implemented at the moment.

    Notes:
    ------
    Internally this function creates RDkit molecule and uses mol_to_spektral(mol, scipy_E = False).
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    assert mol is not None

    if IncludeHs:
        mol = Chem.rdmolops.AddHs(mol)

    can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
    G = mol_to_spektral(mol, 
                        y = y, 
                        u = u, 
                        atom_features = atom_features, 
                        bond_features = bond_features)
    if validate:
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")
    return G


def fasta_to_spektral(fasta, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False,
                    atom_features = ['AtomicNum'], bond_features = ['BondType']):
    """
    Convert FASTA string to spektral.data.graph.Graph.

    Paramters:
    ----------
    fasta : str
        FASTA string that will be converted to rdkit.Chem.rdchem.Mol using 
    """
    mol = Chem.rdmolfiles.MolFromFASTA(fasta.strip())
    assert mol is not None

    if IncludeHs:
        mol = Chem.rdmolops.AddHs(mol)

    can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
    G = mol_to_spektral(mol, 
                        y = y, 
                        u = u,
                        atom_features = atom_features, 
                        bond_features = bond_features)
    if validate:
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")

    return G