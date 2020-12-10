import networkx as nx
import argparse
import multiprocessing
from convert import smiles_to_nx

NUM_PROCESSES = 8

def get_arguments():
    parser = argparse.ArgumentParser(description='Convert an rdkit Mol to nx graph, preserving chemical attributes')
    parser.add_argument('smiles', type=str, help='The input file containing SMILES strings representing an input molecules.')
    parser.add_argument('nx_pickle', type=str, help='The output file containing sequence of pickled nx graphs')
    parser.add_argument('--num_processes', type=int, default=NUM_PROCESSES, help='The number of concurrent processes to use when converting.')
    return parser.parse_args()

def main():
    args = get_arguments()
    i = open(args.smiles)
    p = multiprocessing.Pool(args.num_processes)
    results = p.map(do_all, i.xreadlines())
    o = open(args.nx_pickle, 'w')
    for result in results:
        nx.write_gpickle(result, o)
    o.close()

if __name__ == '__main__':
    main()