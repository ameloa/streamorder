import argparse
import numpy as np
import time
import os
import pandas as pd
import random
random.seed('amel')
from os import path
from enum import Enum
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import sharedctypes
from readdata import get_clean_data
from utils import run_restreaming_greedy
# from plot import draw_figures

parser = argparse.ArgumentParser(description='Run restreaming linear deterministic greedy on a dataset with inputted parameters.')
parser.add_argument('--edge_list', type=str, help='Edge list name')
parser.add_argument('--num_partitions', type=int,  default=16, help='Number of shards in partition')
parser.add_argument('--num_iterations', type=int,  default=10, help='Number of iterations')
parser.add_argument('--num_trials', type=int, default=10,
                    help='Number of trials to run - results are averaged over these trials')
parser.add_argument('--eps', type=float, default=0.0, help='Imbalance paramter')
parser.add_argument('--thresholding', action='store_true', help='Run reLDG with restricting streaming nodes.')
parser.add_argument('--c', type=int, default=0, help='Threshold of gain to include in relocations')
parser.add_argument('--sort_val', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], default=1,
                    help='Random=1, Degree=2, Dec CC=3, Inc CC=4, BFS=5, Ambivalence=6, Amb-degree=7, Gain=8')
parser.add_argument('--no_initialization', action='store_true', help='Run true LDG as iteration 0')
parser.add_argument('--deg_first_step', action='store_true', help='Use degree order for stream 1')
parser.add_argument('--gain_hist', action='store_true', help='Output gains of all nodes across all iterations')
parser.add_argument('--amb_hist', action='store_true', help='Output ambivalences of all nodes across all iterations')
parser.add_argument('--amb_deg_hist', action='store_true', 
                    help='Output ambivalences and degree of all nodes across all iterations')
parser.add_argument('--periodicity', action='store_true', help='Output data for plotting periodicity stackplot')

args = parser.parse_args()

DATA_DIRECTORY = './data'
EDGE_LIST = args.edge_list
FILENAME = os.path.join(DATA_DIRECTORY, EDGE_LIST + '.txt')

# --
edge_dict = {
    'wiki-Vote': 103689, 
    'email-Enron': 367662, 
    'soc-Pokec': 30622564, 
    'com-LiveJournal': 68993773, 
    'com-Orkut': 117185083, 
    'web-NotreDame': 1497134, 
    'web-Stanford': 2312497, 
    'web-Google': 5105039, 
    'web-BerkStan': 7600585,
    'Twitter-2010': 1468364884
}
NUM_EDGES = edge_dict[EDGE_LIST]

NUM_PARTITIONS = args.num_partitions
NUM_ITERATIONS = args.num_iterations
NUM_TRIALS = args.num_trials
EPSILON = args.eps
GAIN_THRESHOLD = args.c

class Sort(Enum):
    #Static
    RANDOM = 1
    DEGREE = 2
    DEC_CC = 3
    INC_CC = 4
    BFS = 5
    #Dynamic
    AMBIVALENCE = 6
    AMBIVALENCE_DEG = 7
    GAIN = 8

SORT = Sort(args.sort_val) #Stream order to use
NO_INITIALIZATION = args.no_initialization #Run true LDG for stream 1
DEGREE_FIRST_STEP = args.deg_first_step #Use degree order for stream 1
THRESHOLDING = args.thresholding #Use degree order for stream 1
GAIN_HISTOGRAM = args.gain_hist #Output gains of all nodes across all iterations
AMB_HISTOGRAM = args.amb_hist #Output ambivalences of all nodes across all iterations
AMB_DEG_HISTOGRAM = args.amb_deg_hist #Output above ^ + degree distribution
PERIODICITY = args.periodicity #Compute and output data for plotting periodicity stack plot

if __name__ == "__main__":
    initializations_file = './reLDGtest/fromBLP/' + EDGE_LIST + '_' + str(NUM_PARTITIONS) + 'shards_inits.txt'
    edges_file = './fromBLP/' + EDGE_LIST + '_edges.txt'

    t = time.time()
#     edges = pd.read_csv(edges_file, delimiter = " ").astype(np.int32)
    if path.exists(edges_file):
        edges = np.loadtxt(edges_file).astype(np.int32)
        NUM_NODES = np.unique(edges).size
        neighborsMap = [[] for node in range(NUM_NODES)]
        node_indices = [[] for node in range(NUM_NODES)]
        for i in range(edges.shape[0]):
            neighborsMap[edges[i,0]].append(edges[i,1])
            node_indices[edges[i,0]].append(i)
    #         neighborsMap[edges.iat[i,0]].append(edges.iat[i,1])
    #         node_indices[edges.iat[i,0]].append(i)
        neighborsMap = np.asarray(neighborsMap)
    else:
        FILENAME = os.path.join('../data/', EDGE_LIST + '.txt')
        neighborsMap, edges = get_clean_data('../data/', FILENAME, NUM_EDGES, shuffle=True)
        NUM_NODES = np.unique(edges).size
        np.savetxt(edges_file, edges.astype(np.int32))
#     initializations = pd.read_csv(initializations_file, delimiter = " ").astype(np.int32)
    if path.exists('./data/' + EDGE_LIST + '_random.txt'):
        node_hash = np.loadtxt('./data/' + EDGE_LIST + '_random.txt')
    else:
        node_hash = np.repeat(np.arange(NUM_NODES).reshape(1,NUM_NODES), NUM_TRIALS, axis=0)
        for i in range(NUM_TRIALS):
            random.shuffle(node_hash[i])
    if not NO_INITIALIZATION:
        initializations = np.loadtxt(initializations_file).astype(np.int32)
    print("%.3f s to load data." % (time.time()-t))

#     fracs = movement = np.zeros((NUM_TRIALS, NUM_ITERATIONS))
#     if AMB_HISTOGRAM:
#         ambivalences = np.zeros((NUM_TRIALS, NUM_ITERATIONS, NUM_NODES))
#     elif AMB_DEG_HISTOGRAM:
#         ambivalences = np.zeros((NUM_TRIALS, NUM_ITERATIONS+1, NUM_NODES))
#     if GAIN_HISTOGRAM:
#         gains = np.zeros((NUM_TRIALS, NUM_ITERATIONS, NUM_NODES))
#     if PERIODICITY:
#         periodicities = np.zeros((NUM_TRIALS, NUM_ITERATIONS, NUM_ITERATIONS))

    fracs = movement = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS)))
    shared_fracs, shared_movement = sharedctypes.RawArray(fracs._type_, fracs), sharedctypes.RawArray(movement._type_, movement)
    if PERIODICITY:
        periodicities = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS, NUM_ITERATIONS)))
        shared_periodicities = sharedctypes.RawArray(periodicities._type_, periodicities)
    if AMB_HISTOGRAM:
        ambivalences = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS, NUM_NODES)))
        shared_ambivalences = sharedctypes.RawArray(ambivalences._type_, ambivalences)
    elif AMB_DEG_HISTOGRAM:
        ambivalences = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS+1, NUM_NODES)))
        shared_ambivalences = sharedctypes.RawArray(ambivalences._type_, ambivalences)
    if GAIN_HISTOGRAM:
        gains = np.ctypeslib.as_ctypes(np.zeros((NUM_TRIALS, NUM_ITERATIONS, NUM_NODES)))
        shared_gains = sharedctypes.RawArray(gains._type_, gains)

    print('\n{}, EPSILON {}'.format(EDGE_LIST, EPSILON))
    def procedure(i):
#     for i in range(NUM_TRIALS):
        fracs = np.ctypeslib.as_array(shared_fracs)
        movement = np.ctypeslib.as_array(shared_movement)
        if PERIODICITY:
            periodicities = np.ctypeslib.as_array(shared_periodicities)
        if AMB_HISTOGRAM | AMB_DEG_HISTOGRAM:
            ambivalences = np.ctypeslib.as_array(shared_ambivalences)
        if GAIN_HISTOGRAM:
            gains = np.ctypeslib.as_array(shared_gains)

        t = time.time()
        edge_score_values, nodes_moving, periodicity, ambivalence, gain = run_restreaming_greedy(i, EDGE_LIST, edges, neighborsMap, node_indices, NUM_NODES, hashMap=node_hash[i], num_partitions=NUM_PARTITIONS, num_iterations=NUM_ITERATIONS, eps=EPSILON, degree_first_step=DEGREE_FIRST_STEP, sort_type_value=SORT.value, periodicity=PERIODICITY, gain_histogram=GAIN_HISTOGRAM, amb_histogram=AMB_HISTOGRAM, amb_deg_histogram=AMB_DEG_HISTOGRAM, thresholding=THRESHOLDING, c=GAIN_THRESHOLD) if NO_INITIALIZATION else run_restreaming_greedy(i, EDGE_LIST, edges, neighborsMap, node_indices, NUM_NODES, hashMap=node_hash[i], num_partitions=NUM_PARTITIONS, num_iterations=NUM_ITERATIONS, eps=EPSILON, degree_first_step=DEGREE_FIRST_STEP, sort_type_value=SORT.value, periodicity=PERIODICITY, gain_histogram=GAIN_HISTOGRAM, amb_histogram=AMB_HISTOGRAM, amb_deg_histogram=AMB_DEG_HISTOGRAM, thresholding=THRESHOLDING, c=GAIN_THRESHOLD, assignments=initializations[i,:])
        t_final = time.time()-t
        fracs[i,:] = edge_score_values
        movement[i,:] = nodes_moving
        if PERIODICITY:
            periodicities[i,:,:] = periodicity
        if AMB_HISTOGRAM | AMB_DEG_HISTOGRAM:
            ambivalences[i,:,:] = ambivalence
        if GAIN_HISTOGRAM:
            gains[i,:,:] = gain
            
        string = (str(i) + ": Completed " + SORT.name + " reLDG. (%.3f s)") % t_final
        print(string)

    p = Pool()
    res = p.map(procedure, np.arange(NUM_TRIALS))
    fracs = np.ctypeslib.as_array(shared_fracs)
    movement = np.ctypeslib.as_array(shared_movement)
    if PERIODICITY:
        periodicities = np.ctypeslib.as_array(shared_periodicities)
    if AMB_HISTOGRAM | AMB_DEG_HISTOGRAM:
        ambivalences = np.ctypeslib.as_array(shared_ambivalences)
    if GAIN_HISTOGRAM:
        gains = np.ctypeslib.as_array(shared_gains)

    finalFrac = np.mean(fracs[:,-1])
    print(' - Final fractions, {}, {}, EPSILON = {}:'.format(EDGE_LIST, SORT.name, EPSILON))
    print('     - Edge Frac %.3f' % finalFrac)
#     print(GAIN_THRESHOLD, finalFrac)
    
    
    if NO_INITIALIZATION:
        output_result = './results/' + EDGE_LIST + '_' + str(NUM_ITERATIONS) + 'iters_' + str(NUM_PARTITIONS) + 'shards_' + SORT.name + '_DEGREEFIRST_trueLDG' if DEGREE_FIRST_STEP else './results/' + EDGE_LIST + '_' + str(NUM_ITERATIONS) + 'iters_' + str(NUM_PARTITIONS) + 'shards_' + SORT.name + '_trueLDG'
    else:
        output_result = './results/' + EDGE_LIST + '_' + str(NUM_ITERATIONS) + 'iters_' + str(NUM_PARTITIONS) + 'shards_' + SORT.name + '_DEGREEFIRST' if DEGREE_FIRST_STEP else './results/' + EDGE_LIST + '_' + str(NUM_ITERATIONS) + 'iters_' + str(NUM_PARTITIONS) + 'shards_' + SORT.name
        
    if EPSILON == 0.0:
        np.savetxt(output_result + '_eps0_fracs.txt', fracs)
        np.savetxt(output_result + '_eps0_movers.txt', movement)
        if PERIODICITY:
            np.save(output_result + '_eps0_periodicity', periodicities)
        if GAIN_HISTOGRAM:
            np.save(output_result + '_eps0_gain', gains)
        if AMB_HISTOGRAM:
            np.save(output_result + '_eps0_ambivalence', ambivalences)
        if AMB_DEG_HISTOGRAM:
            np.save(output_result + '_eps0_amb_deg', ambivalences)
    else:
        np.savetxt(output_result + '_eps05_fracs.txt', fracs)
        np.savetxt(output_result + '_eps05_movers.txt', movement)
        if PERIODICITY:
            np.save(output_result + '_eps05_periodicity', periodicities)
        if GAIN_HISTOGRAM:
            np.save(output_result + '_eps05_gain', gains)
        if AMB_HISTOGRAM:
            np.save(output_result + '_eps05_ambivalence', ambivalences)
        if AMB_DEG_HISTOGRAM:
            np.save(output_result + '_eps05_amb_deg', ambivalences)
