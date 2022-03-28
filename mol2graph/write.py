from rdkit import Chem

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# def write_fasta(filepath, fasta_dict):
#     with open(filepath, "w") as outfile:
#         for key in fasta_dict.keys():
#             outfile.write(">" + key + "\n" + fasta_dict[key] + "\n")
#     return None

def write_fasta(filepath, fasta_dict):
    records = []
    for key in fasta_dict.keys():
        records.append(SeqRecord(Seq(fasta_dict[key]), id=key, description = ""))
    with open(filepath, "w") as outfile:
        SeqIO.write(records, outfile, "fasta")
    return None


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