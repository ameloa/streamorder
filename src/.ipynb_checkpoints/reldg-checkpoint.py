import sys
sys.path.append('..')
import numpy as np
from shardmap import *

from os import path
from nodeorders import *
from alg import linear_deterministic_greedy
from shp import *

def get_edge_indices(order, node_indices):
    # returns indices of correct sort order in edges list
    sorted_edge_indices = []
    for node in order:
        sorted_edge_indices.extend(node_indices[int(node)])
    return np.asarray(sorted_edge_indices, dtype=np.int32)

def compute_gains(numNodes, neighborsMap, assignments=None):
    gains = np.zeros(numNodes)
    for node in range(numNodes):
        neighbor_shards = [assignments[v] for v in neighborsMap[node]]
        unique, counts = np.unique(np.array(neighbor_shards), return_counts = True)
        indices = np.nonzero(unique != assignments[node])
        best = max(counts[indices]) if len(counts[indices]) != 0 else 0
        my_neighbor_count = np.sum(neighbor_shards == assignments[node])
        gain = best - my_neighbor_count
        gains[node] = gain
    return gains

'''
    Implementation of Restreamed Linear Deterministic Greedy.
    For implementation details, see the original papers by Nishimura & Ugander and Stanton & Kliot:
    https://stanford.edu/~jugander/papers/kdd13-restream.pdf
    https://www.microsoft.com/en-us/research/wp-content/uploads/2012/08/kdd325-stanton.pdf
'''
def reldg(numNodes, edges, node_indices, neighborsMap, numShards=16, numIterations=10, epsilon=0.0, return_periodicity=False, return_orders=False, thresholding=False, c=-np.inf, version='random', bfs_order = None, ccs = None):
    edgeFracs = []
#     movers = []
    orders = []
    shardMap = None
        
    if return_periodicity:
        print('Initializing structures for collecting periodicity data...')
        periodicity = np.zeros((numIterations, numIterations))
        assignmentHistory = np.full((numNodes, numShards), np.inf)
    else:
        periodicity = None
    
    print('Stream order: ', version)
    print('Computing static stream order, or first stream of dynamic order...')
    hashMap = np.arange(numNodes)
    random.shuffle(hashMap)       
    if version == 'random':
        order = hashMap
        indices = get_edge_indices(order, node_indices)
    if version == 'cc':
        if ccs is None:
            print('Must pass local clustering coefficients into function.')
            return
        else:
            order = np.lexsort((hashMap, -ccs))
            indices = get_edge_indices(order, node_indices)
    if version == 'bfs':
        if bfs_order is None:
            print('Must pass bfs ordering into function.')
            return
        else:
            indices = get_edge_indices(bfs_order, node_indices)
    if version == 'degree' or version == 'ambivalence' or version == 'gain':
        degrees = np.array([len(neighborsMap[node]) for node in range(numNodes)])
        order = np.lexsort((hashMap, -degrees)) # first stream in degree order
        indices = get_edge_indices(order, node_indices)
        

    print('Beginning reLDG...')
    for i in range(numIterations):
        istr = str(i)+ ": "
        
        # for dynamic orders, or if gain thresholding is used
        if version == 'ambivalence' and (shardMap is not None):
            gains = compute_gains(numNodes, neighborsMap, assignments=shardMap)
            ambivalence = -np.abs(np.array(gains))
            if return_orders and (i == 1 or i == (numIterations-1)):
                orders.append(ambivalence)
            stream_indices = get_edge_indices(np.lexsort((hashMap, ambivalence)), node_indices)
        elif version == 'gain' and (shardMap is not None):
            gains = compute_gains(numNodes, neighborsMap, assignments=shardMap)
            if return_orders and (i == 1 or i == numIterations-1):
                orders.append(gains)
            stream_indices = get_edge_indices(np.lexsort((hashMap, gains)), node_indices)     
        elif thresholding and (shardMap is not None):
                gains = compute_gains(numNodes, neighborsMap, assignments=shardMap)
                movingNodes = np.nonzero(gains>c)[0]
                stayingNodes = np.nonzero(gains<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], -gains[movingNodes]))], node_indices)
                currentHist = np.zeros(numShards, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[shardMap[node]] += 1
        else:
            stream_indices = indices
                
        # run one iteration.
        # if thresholding, pass in histogram of incumbent assignments.
        if thresholding and (shardMap is not None):
            _, shardMap = linear_deterministic_greedy(edges, stream_indices, numNodes, numShards, shardMap, currentHist, epsilon)
        else:
            _, shardMap = linear_deterministic_greedy(edges, stream_indices, numNodes, numShards, shardMap, None, epsilon)
#         movers.append(num_nodes_moving)

        # evaluate partition quality
        internal, external = shardmap_evaluate(shardMap, numNodes, neighborsMap)
        edgeFracs.append(float(internal)/(internal+external))
        print(istr + 'Internal edge fraction = ', float(internal)/(internal+external))
        if return_periodicity:
            assert(len(shardMap) == numNodes)
            for node in range(numNodes):
                period = i - assignmentHistory[node, shardMap[node]] if assignmentHistory[node, shardMap[node]] != np.inf else 0
                periodicity[int(period), i] += 1
                assignmentHistory[node, shardMap[node]] = i

    return edgeFracs, orders, periodicity
