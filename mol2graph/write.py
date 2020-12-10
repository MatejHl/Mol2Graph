from rdkit import Chem

def smiles_to_sdf(filename, mols):
    """
    Write mols according to order in pairs (potentially twice) to SDF:
    INEFFICIENT
    TO DO: Add and read IDs (CID)

    Parameters:
    -----------
    filename : str
        file name

    mols : pandas.Series
        SMILES in pandas series with PubChem CID as index.
    """
    # pylint: disable=no-member
    writer = Chem.SDWriter(filename)
    for cid, smiles in mols.iteritems():
        m = Chem.MolFromSmiles(smiles)
        m.SetProp('CID', str(cid))
        writer.write(m)
    writer.flush()
    writer.close()