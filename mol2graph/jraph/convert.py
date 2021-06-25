import jax
from jax import numpy as jnp
import jraph

from rdkit import Chem

def mol_to_jraph(mol, u = None, atom_features = [], bond_features = []):
    """
    See 'rdkit.Chem.rdchem.Atom' for all possible attributes of nodes and
    'rdkit.Chem.rdchem.Bond' for all possible attributes of edges 
    at https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

    Paramters:
    ----------
    mol : rdkit.Chem.rdchem.Mol

    atom_features : list
        node attributes. Can contain: 
            [AtomicNum
             FormalCharge
             ChiralTag
             Hybridization
             NumExplicitHs
             NumImplicitHs
             TotalNumHs
             ExplicitValence
             ImplicitValence
             TotalValence
             IsInRing
             Mass
             NumRadicalElectrons
             IsAromatic]

    bond_features : list
        Can contain:
            [BondType
             IsAromatic
             IsConjugated
             Stereo
             ValenceContrib
             IsInRing]

    u : np.array or convertable to jax.numpy.array
        graph level features for a given graph.

    Returns:
    --------
    G : jraph.GraphsTuple

    Notes:
    ------
    Categorical features in X and E are encoded by rdkit.Chem.rdchem standard numbering and
    mapping from numbers to names can be accessed from CHIRAL_TAG, HYBRIDIZATION and BOND_TYPE
    objects (dictionaries) from rdkit.
    """
    X = []
    begin = []
    end = []
    E = []
    for atom in mol.GetAtoms():
        props = []
        for prop in atom_features:
            if   prop == 'AtomicNum':           props.append(atom.GetAtomicNum())
            elif prop == 'FormalCharge':        props.append(atom.GetFormalCharge())
            elif prop == 'ChiralTag':           props.append(int(atom.GetChiralTag())) # TODO: Check
            elif prop == 'Hybridization':       props.append(int(atom.GetHybridization()))
            elif prop == 'NumExplicitHs':       props.append(atom.GetNumExplicitHs())
            elif prop == 'NumImplicitHs':       props.append(atom.GetNumImplicitHs())
            elif prop == 'TotalNumHs':          props.append(atom.GetTotalNumHs())
            elif prop == 'ExplicitValence':     props.append(atom.GetExplicitValence())
            elif prop == 'ImplicitValence':     props.append(atom.GetImplicitValence())
            elif prop == 'TotalValence':        props.append(atom.GetTotalValence())
            elif prop == 'IsInRing':            props.append(atom.IsInRing())
            elif prop == 'Mass':                props.append(atom.GetMass())
            elif prop == 'NumRadicalElectrons': props.append(atom.GetNumRadicalElectrons())
            elif prop == 'IsAromatic':          props.append(int(atom.GetIsAromatic()))
            else:
                raise ValueError('atom feature {} is unavailable in RDkit'.format(prop))
        X.append(props)     
   
    for bond in mol.GetBonds():
        begin.append(bond.GetBeginAtomIdx())
        end.append(bond.GetEndAtomIdx())
        props = []
        for prop in bond_features:
            if   prop == 'BondType':        props.append(int(bond.GetBondType()))
            elif prop == 'IsAromatic':      props.append(bond.GetIsAromatic())
            elif prop == 'IsConjugated':    props.append(bond.GetIsConjugated())
            elif prop == 'Stereo':          props.append(bond.GetStereo())
            # elif prop == 'ValenceContrib':  props.append(bond.GetValenceContrib()) # TODO: This is specific for Atom-Bond pair.
            elif prop == 'IsInRing':        props.append(bond.IsInRing())
            else:
                raise ValueError('bond feature {} is unavailable in RDkit'.format(prop))
        E.append(props)    

    n_node = jnp.array([len(X)])
    n_edge = jnp.array([2*len(E)])
    x = jnp.array(X, dtype = jnp.float32)
    senders = jnp.array(begin + end, dtype = jnp.int32)
    receivers = jnp.array(end + begin, dtype = jnp.int32)
    e = jnp.array(E + E, dtype = jnp.float32)
    if u is not None:
        u = jax.tree_map(lambda x: jnp.array(x), u)

    if len(e.shape) != 2:
        print('WARNING: Molecule with no bonds and {} atoms'.format(len(X)))

    assert len(e.shape) == 2 or n_edge == 0

    G = jraph.GraphsTuple(nodes = x,
                        edges = e,
                        receivers = receivers,
                        senders = senders,
                        globals = u,
                        n_node = n_node,  
                        n_edge = n_edge)
    return G


def smiles_to_jraph(smiles, u = None, validate=False, IncludeHs = False,
                        atom_features = ['AtomicNum'], bond_features = ['BondType']):
    """
    Convert SMILES string to jraph.GraphsTuple.

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
    Internally this function creates RDkit molecule and uses mol_to_jraph.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    assert mol is not None

    if IncludeHs:
        mol = Chem.rdmolops.AddHs(mol)

    if validate:
        can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")
    
    G = mol_to_jraph(mol, 
                    u = u, 
                    atom_features = atom_features, 
                    bond_features = bond_features)
    return G



if __name__ == '__main__':
    smiles = 'C/C/1=C/CC/C(=C\[C@H]2[C@H](C2(C)C)CC1)/C'

    G1 = smiles_to_jraph(smiles, u = None, atom_features = ['AtomicNum'], bond_features = ['BondType'], IncludeHs = True)

    print(G1)

    smiles = 'F/C=C\F'

    G2 = smiles_to_jraph(smiles, u = None, atom_features = ['AtomicNum'], bond_features = ['BondType'])

    smiles = 'CC(C)C(C(=N)O)N=C(C1CCCN1C(=O)C(CCCCN)N=C(CN=C(C(CC2=CNC3=CC=CC=C32)N=C(C(CCCNC(=N)N)N=C(C(CC4=CC=CC=C4)N=C(C(CC5=CN=CN5)N=C(C(CCC(=O)O)N=C(C(CCSC)N=C(C(CO)N=C(C(CC6=CC=C(C=C6)O)N=C(C(CO)N=C(C)O)O)O)O)O)O)O)O)O)O)O)O'

    G3 = smiles_to_jraph(smiles, u = None, atom_features = ['AtomicNum'], bond_features = ['BondType'])

    G = jraph.batch([G1, G2, G3])

    print(G)

    G = jraph.unbatch(G)[0]

    import networkx as nx
    _G=nx.DiGraph()
    _G.add_nodes_from(list(range(G.nodes.shape[0])))
    _G.add_edges_from(list(zip(G.senders, G.receivers)))

    import matplotlib.pyplot as plt

    nx.draw(_G)
    plt.show()
