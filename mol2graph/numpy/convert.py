import numpy as np

from rdkit import Chem

def mol_to_numpy(mol, y = None, u = None, atom_features = [], bond_features = []):
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
    x : 
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

    a : np.array
        graph Adjecency matrix.
    
    e : np.array
        edge attributes.
        E is stored as a numpy array where elements are ordered by (row_idx, col_idx) (so similar 
        to scipy.sparse.crs_matrix) and adjecancy matrix is used to extraxt which to which edge the
        data corresponds.
        Can contain:
            [BondType
             IsAromatic
             IsConjugated
             Stereo
             ValenceContrib
             IsInRing]

    y : np.array
        labels for a given graph. Only returning input y.
    
    u : np.array
        graph level features for a given graph. Only returning input y.

    Notes:
    ------
    Categorical features in X and E are encoded by rdkit.Chem.rdchem standard numbering and
    mapping from numbers to names can be accessed from CHIRAL_TAG, HYBRIDIZATION and BOND_TYPE
    objects (dictionaries).
    """
    A = Chem.GetAdjacencyMatrix(mol)
    assert np.all(np.sum(A, axis=1) > 0)
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
        # NOTE: atom_features below were used before in Odorant_perception.
        # ['AtomicNum', 'FormalCharge', 'ChiralTag', 'Hybridization', 
        # 'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic']
        # X.append([atom.GetAtomicNum(),          # atomic_num
        #         atom.GetFormalCharge(),         # formal_charge
        #         int(atom.GetChiralTag()),       # chiral_tag
        #         int(atom.GetHybridization()),   # hybridization 
        #         # atom.GetNumExplicitHs(),      # num_explicit_hs
        #         atom.GetNumImplicitHs(),        # num_implicit_hs
        #         # atom.GetTotalNumHs(),         # num_explicit_hs + num_implicit_hs
        #         atom.GetExplicitValence(),
        #         # atom.GetImplicitValence(),
        #         # atom.GetTotalValence(),
        #         # atom.IsInRing(),                   
        #         atom.GetMass(),                 # mass
        #         # atom.GetNumRadicalElectrons(),
        #         int(atom.GetIsAromatic()), # is_aromatic
        #         ])  
   
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

        # NOTE: bond_features below were used before in Odorant_perception.
        # ['BondType', 'IsAromatic']
        # E.append([int(bond.GetBondType()),      # bond_type
        #         bond.GetIsAromatic(),         
        #         # bond.GetIsConjugated(),
        #         # bond.GetStereo(),             # stereo_configuration
        #         # bond.GetValenceContrib(),      # contrib_to_valance
        #         # bond.IsInRing(),
        #         ])     

    _E = list(zip(begin + end, end + begin, E + E))
    _E.sort(key=lambda ele: (ele[0], ele[1]))
    E = np.array([dat[2] for dat in _E], dtype=np.float32)

    assert len(E.shape) == 2

    # data = list(itertools.chain.from_iterable(E))
    # E = scipy.sparse.csr_matrix((data + data, (begin + end, end + begin)), shape = Chem.GetAdjacencyMatrix(mol).shape)
    # E.sort_indices()

    x = np.array(X, dtype = np.float32)
    a = A.astype(np.float32)
    e = E
    
    G = (x, a, e, y, u)
    return G


def smiles_to_numpy(smiles, y = None, u = None, validate=False, scipy_E = False, IncludeHs = False,
                    atom_features = ['AtomicNum'], bond_features = ['BondType']):
    """
    Convert SMILES string to list of numpy arrays.

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
        
    if validate:
        # can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")

    G = mol_to_numpy(mol, 
                    y = y, 
                    u = u,
                    atom_features = atom_features, 
                    bond_features = bond_features)
    G = [s for s in G if s is not None]
    return G