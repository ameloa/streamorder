import argparse
import numpy as np
import time
import os
from shardmap import shardmap_random_init
from enum import Enum
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import sharedctypes
from socialhash import *

parser = argparse.ArgumentParser(description='Run restreaming linear deterministic greedy on a dataset with inputted parameters.')
parser.add_argument('--edge_list', type=str, help='Edge list name')
parser.add_argument('--num_partitions', type=int,  default=16, help='Number of shards in partition')
parser.add_argument('--num_iterations', type=int,  default=10, help='Number of iterations')
parser.add_argument('--num_trials', type=int, default=10,
                    help='Number of trials to run - results are averaged over these trials')
parser.add_argument('--type_val', type=int, choices=[1,2], default=2, help='Probabilistic=1, Deterministic=2')
parser.add_argument('--sort_val', type=int, choices=[1,2,3], default=1, help='None=1, Gain=2, KL gain=3')
parser.add_argument('--thresholding', action='store_true', help='Run SHP thresholding nodes.')
parser.add_argument('--c', type=int, default=0, help='Threshold of gain to include in relocations')
parser.add_argument('--periodicity', action='store_true', help='Output data for plotting periodicity stackplot')

args = parser.parse_args()

DATA_DIRECTORY = './data'
EDGE_LIST = args.edge_list
FILENAME = os.path.join(DATA_DIRECTORY, EDGE_LIST + '.txt')

# NOT NEEDED:
# --
# edge_dict = {
#     'wiki-Vote': 103689, 
#     'email-Enron': 367662, 
#     'soc-Pokec': 30622564, 
#     'com-LiveJournal': 68993773, 
#     'com-Orkut': 117185083, 
#     'web-NotreDame': 1497134, 
#     'web-Stanford': 2312497, 
#     'web-Google': 5105039, 
#     'web-BerkStan': 7600585
# }
# NUM_EDGES = edge_dict[EDGE_LIST]

NUM_PARTITIONS = args.num_partitions
NUM_ITERATIONS = args.num_iterations
NUM_TRIALS = args.num_trials

class Type(Enum):
    PROBABILISTIC = 1
    DETERMINISTIC = 2
    
TYPE = Type(args.type_val)

class Sort(Enum):
    NONE = 1
    GAIN = 2
    KL_GAIN = 3

if TYPE.value == 2:
    SORT = Sort(args.sort_val)
else:
    SORT = Sort(1)
    
THRESHOLD = float(args.thresholding)
GAIN_THRESHOLD = float(args.c)
    
PERIODICITY = args.periodicity

if __name__ == "__main__":
    initializations_file = './reLDGtest/fromBLP/' + EDGE_LIST + '_' + str(NUM_PARTITIONS) + 'shards_inits.txt'
    edges_file = './reLDGtest/fromBLP/' + EDGE_LIST + '_edges.txt'

    print("Normalizing data (or loaded pre-computed)")
    t = time.time()
    edges = np.loadtxt(edges_file).astype(np.int32)
    nodes = np.unique(edges).astype(np.int32)
    NUM_NODES = len(nodes)
    neighborsMap = [[] for node in range(NUM_NODES)]
    node_indices = [[] for node in range(NUM_NODES)]
    for i in range(edges.shape[0]):
        neighborsMap[edges[i,0]].append(edges[i,1])
        node_indices[edges[i,0]].append(i)
    neighborsMap = np.asarray(neighborsMap)
    if os.path.exists(initializations_file):
        initializations = np.loadtxt(initializations_file).astype(np.int32)
    else:
        initializations = np.array([shardmap_random_init(nodes, NUM_PARTITIONS) for _ in range(10)]).astype(np.int32)
    print("%.3f s to load data. \n" % (time.time()-t))

#     fracs = movers = np.zeros((NUM_TRIALS, NUM_ITERATIONS))
#     if PERIODICITY:
#         periodicities = np.zeros((NUM_TRIALS, NUM_ITERATIONS+1, NUM_ITERATIONS))
        
    fracs = movers = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS)))
    shared_fracs, shared_movers = sharedctypes.RawArray(fracs._type_, fracs), sharedctypes.RawArray(movers._type_, movers)
    if PERIODICITY:
        periodicities = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS+1, NUM_ITERATIONS)))
        shared_periodicities = sharedctypes.RawArray(periodicities._type_, periodicities)
        
    print('\n{}'.format(EDGE_LIST))
    def procedure(i):
#     for i in range(NUM_TRIALS):
        fracs = np.ctypeslib.as_array(shared_fracs)
        movers = np.ctypeslib.as_array(shared_movers)
        if PERIODICITY:
            periodicities = np.ctypeslib.as_array(shared_periodicities)

        t = time.time()
        shpFracs, shpMovers, shpPeriodicity = SocialHash(np.arange(NUM_NODES), neighborsMap, NUM_PARTITIONS, NUM_ITERATIONS, 0.0, PERIODICITY, initializations[i,:], TYPE.value, SORT.value, THRESHOLD, GAIN_THRESHOLD)
        t_final = time.time()-t
        fracs[i,:] = shpFracs
        movers[i,:] = shpMovers
        if PERIODICITY:
            periodicities[i,:,:] = shpPeriodicity
        print((str(i) + ': Completed social hash partitioner. (%.3f s)') % t_final)

    p = Pool()
    res = p.map(procedure, np.arange(NUM_TRIALS))
    fracs = np.ctypeslib.as_array(shared_fracs)
    movers = np.ctypeslib.as_array(shared_movers)
    if PERIODICITY:
        periodicities = np.ctypeslib.as_array(shared_periodicities)

    finalFrac = np.sum(fracs[:,-1])/NUM_TRIALS
    print(' - Final fractions:')
    if THRESHOLD:
        print('     c=%d - Edge Frac %.3f' % (GAIN_THRESHOLD, finalFrac))
    else:
        print('     - Edge Frac %.3f' % (finalFrac))
#     print('            Movers: \n{}'.format(movers))
#     print(GAIN_THRESHOLD, finalFrac)
    output_result = './results/shp/' + EDGE_LIST + '_' + str(NUM_ITERATIONS) + 'iters_' + str(NUM_PARTITIONS) + 'shards_' + TYPE.name + '_' + SORT.name + '_sort'
    np.savetxt(output_result + '_eps0_fracs.txt', fracs)
    np.savetxt(output_result + '_eps0_movers.txt', movers)
    if PERIODICITY:
        np.save(output_result + '_eps0_periodicity', periodicities)
