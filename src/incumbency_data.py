import sys
import os
import argparse
import numpy as np

import readdata
from blp import blp
from shp import shp
from reldg import reldg

parser = argparse.ArgumentParser(description='Run balanced label propagation on a dataset with inputted parameters.')
parser.add_argument('--edge_list', type=str, help='Edge list name')
parser.add_argument('--num_partitions', type=int,  default=16, help='Number of shards in partition')
parser.add_argument('--num_iterations', type=int,  default=10, help='Number of iterations')
parser.add_argument('--eps', type=float,  default=0.0, help='Imbalance paramter')
parser.add_argument('--c', type=int, default=0, help='Threshold of gain to include in relocations')

args = parser.parse_args()
DATA_DIRECTORY = '../data/'
EDGE_LIST = args.edge_list
FILENAME = os.path.join(DATA_DIRECTORY, EDGE_LIST + '.txt')

edge_dict = {
    'wiki-Vote': 103689, 
    'email-Enron': 367662, 
    'soc-Pokec': 30622564,
    'com-LiveJournal': 68993773,
    'com-Orkut': 117185083, 
    'web-NotreDame': 1497134, 
    'web-Stanford': 2312497, 
    'web-Google': 5105039, 
    'web-BerkStan': 7600595
}
NUM_EDGES = edge_dict[EDGE_LIST]

NUM_PARTITIONS = args.num_partitions
NUM_ITERATIONS = args.num_iterations
EPSILON = args.eps
GAIN_THRESHOLD = args.c

if __name__ == "__main__":
    # supresses printing
    sys.stdout = open(os.devnull, 'w')

    print('Reading edge list in to memory...', end =" ")
    neighbors_map, num_nodes, node_indices, edges = readdata.get_clean_data(FILENAME, NUM_EDGES)
    print('Done.')
    if not os.path.exists(DATA_DIRECTORY + EDGE_LIST + '_edges.txt'):
        np.savetxt(DATA_DIRECTORY + EDGE_LIST + '_edges.txt', edges.astype(np.int32))
    initialization_file = DATA_DIRECTORY + EDGE_LIST + '_' + str(NUM_PARTITIONS) + 'shards_init.txt'
    if os.path.exists(initialization_file):
        initialization = np.loadtxt(initialization_file).astype(np.int32)
    else:
        initialization = None

    initialization, blpEdgeFracs, _ = blp(num_nodes, 
                                          neighbors_map, 
                                          NUM_PARTITIONS, 
                                          NUM_ITERATIONS, 
                                          EPSILON,
                                          c = GAIN_THRESHOLD,
                                          initialization = initialization)
    if not os.path.exists(initialization_file):
        np.savetxt(initialization_file, initialization)
    klshpEdgeFracs, _ = shp(num_nodes,
                            neighbors_map,
                            NUM_PARTITIONS,
                            NUM_ITERATIONS,
                            c = GAIN_THRESHOLD,
                            initialization = np.loadtxt(initialization_file).astype(np.int32),
                            version = 'KL')
    reldgEdgeFracs, _, _ = reldg(num_nodes, 
                                 edges, 
                                 node_indices, 
                                 neighbors_map,
                                 NUM_PARTITIONS, 
                                 NUM_ITERATIONS,
                                 EPSILON,
                                 thresholding = True,
                                 c = GAIN_THRESHOLD,
                                 version = 'random')

    # enables printing again
    sys.stdout = sys.__stdout__
    print(GAIN_THRESHOLD, blpEdgeFracs[-1], klshpEdgeFracs[-1], reldgEdgeFracs[-1])
