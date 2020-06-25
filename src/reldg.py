import sys
sys.path.append('..')
import numpy as np
from shardmap import *

from os import path
from nodeorders import *
# from collections import deque
from alg import linear_deterministic_greedy
from shp import *

# def score(num_nodes, neighborsMap, num_partitions, assignment, edges):
#     """Compute the score given an assignment of vertices.

#     N nodes are assigned to clusters 0 to K-1.

#     assignment: Vector where N[i] is the cluster node i is assigned to.
#     edges: The edges in the graph, assumed to have one in each direction

#     Returns: (total wasted bin space, ratio of edges cut)
#     """

#     left_edge_assignment = assignment.take(edges[:,0])
#     right_edge_assignment = assignment.take(edges[:,1])
#     match = (left_edge_assignment == right_edge_assignment).sum()
#     int_ratio = float(match) / float(len(edges))
#     return int_ratio

def get_edge_indices(order, node_indices):
    # returns indices of correct sort order in edges
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

# def compute_clustering_coefficients(edge_list, num_nodes, adjacency_list, node_hash=None):
#     import_path = './data/' + edge_list + '_ccs.txt'
#     if path.exists(import_path):
#         cc = np.loadtxt(import_path)
#         assert(len(cc) == num_nodes)
#     else:
#         cc = np.zeros(num_nodes)
#         for node in range(num_nodes):
#             if len(adjacency_list[node]) < 2:
#                 continue
#             deg = len(adjacency_list[node])
#             triangles = 0
#             neighbors = adjacency_list[node]
#             for i in range(len(neighbors)):
#                 for j in range(i+1, len(neighbors)):
#                     if adjacency_list[node][j] in adjacency_list[adjacency_list[node][i]]:
#                         triangles += 1
#             cc[node] = 2.0*float(triangles)/float(deg*(deg-1))
#             np.savetxt(import_path, cc)
#     return cc

# def bfs_order(start, neighborsMap, visited, hashMap):
#     q = deque()
#     q.append(start)
#     visited[start] = True
#     bfs = []
#     count = 0
#     while len(q) > 0:
#         curr_node = q.popleft()
#         bfs.append(curr_node)
#         count += 1
#         inds = np.argsort(hashMap[neighborsMap[curr_node]])
#         for i in inds:
#             neighbor = neighborsMap[curr_node][i]
#             if not visited[neighbor]:
#                 visited[neighbor] = True
#                 q.append(neighbor)
#     return bfs, count

# def bfs_disconnected(num_nodes, neighborsMap, degrees, hashMap=None):
#     if hashMap is None:
#         hashMap = np.arange(num_nodes)
#     nodes = np.argsort(-degrees)
#     visited = np.full(num_nodes, False)
#     order = []
#     connected_components = []
#     for i in range(len(nodes)):
#         if not visited[nodes[i]]:
#             curr_order, num_reached = bfs_order(nodes[i], neighborsMap, visited, hashMap)
#             order.extend(curr_order)
#             connected_components.append(num_reached)
#     assert(len(order) == num_nodes)
#     assert(sum(connected_components) == num_nodes)
#     return order, connected_components
        
'''
    Implementation of Restreamed Linear Deterministic Greedy.
    For implementation details, see the original papers by Nishimura & Ugander and Stanton & Kliot:
    https://stanford.edu/~jugander/papers/kdd13-restream.pdf
    https://www.microsoft.com/en-us/research/wp-content/uploads/2012/08/kdd325-stanton.pdf
'''
def reldg(numNodes, edges, node_indices, neighborsMap, numShards=16, numIterations=10, epsilon=0.0, return_periodicity=False, thresholding=False, c=-np.inf, version='random'):
    edgeFracs = []
    movers = []
    shardMap = None
    
    if version == 'ambivalence':
        degrees = np.array([len(neighborsMap[node]) for node in range(numNodes)])
    
    if return_periodicity:
        print('Initializing structures for collecting periodicity data...')
        periodicity = np.zeros((numIterations, numIterations))
        assignmentHistory = np.full((numNodes, numShards), np.inf)
    else:
        periodicity = None
        
    hashMap = np.arange(numNodes)
    random.shuffle(hashMap)       
    if version == 'random':
        order = hashMap
        indices = get_edge_indices(order, node_indices)
    if version == 'ambivalence':
        order = np.lexsort((hashMap, -degrees)) # first stream in degree order
        indices = get_edge_indices(order, node_indices)
#     if sort_type_value == 3:
#         ccs = compute_clustering_coefficients(edge_list, num_nodes, neighborsMap)
#         order = np.lexsort((hashMap, -ccs))
#         indices = get_edge_indices(order, node_indices)
#     if sort_type_value == 4:
#         ccs = compute_clustering_coefficients(edge_list, num_nodes, neighborsMap)
#         order = np.lexsort((hashMap, ccs))
#         indices = get_edge_indices(order, node_indices)
#     if sort_type_value == 5:
#         bfs_file = './data/' + edge_list + '_bfs.txt'
#         order = np.loadtxt(bfs_file)[i] if path.exists(bfs_file) else bfs_disconnected(num_nodes, neighborsMap, degrees, hashMap)
#         if not path.exists(bfs_file):
#             np.savetxt(bfs_file, order)
#         indices = get_edge_indices(order, node_indices)
    print('Beginning reLDG, version ' + version + '...')
    for i in range(numIterations):
        istr = str(i)+ ": "
        if i == 0:
            stream_indices = indices
        elif version == 'ambivalence' and (shardMap is not None):
            gains = compute_gains(numNodes, neighborsMap, assignments=shardMap)
            ambivalence = -np.abs(np.array(gains))
            if thresholding:
                movingNodes = np.nonzero(gains>c)[0]
                stayingNodes = np.nonzero(gains<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], ambivalence[movingNodes]))], node_indices)
                currentHist = np.zeros(num_partitions, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[shardMap[node]] += 1
            else:
                stream_indices = get_edge_indices(np.lexsort((hashMap, ambivalence)), node_indices)
        else:
            if thresholding and (shardMap is not None):
                gains = compute_gains(numNodes, neighborsMap, assignments=shardMap)
                movingNodes = np.nonzero(gains>c)[0]
                stayingNodes = np.nonzero(gains<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], -gains[movingNodes]))], node_indices)
                currentHist = np.zeros(numShards, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[shardMap[node]] += 1
            else:
                stream_indices = indices
                
#         if (gain_histogram or amb_histogram or amb_deg_histogram) and (assignments is not None):
#             gain = compute_gains(num_nodes, neighborsMap, normed=True, assignments=assignments)
#             if amb_histogram or amb_deg_histogram:
#                 ambivalences[i,:] = -np.abs(gain)
#             if gain_histogram:
#                 gain[gain < 0] = 0
#                 gains[i,:] = gain

        if thresholding and (assignments is not None):
            num_nodes_moving, shardMap = linear_deterministic_greedy(edges, stream_indices, numNodes, numShards, shardMap, currentHist, epsilon)
        else:
            num_nodes_moving, shardMap = linear_deterministic_greedy(edges, stream_indices, numNodes, numShards, shardMap, None, epsilon)
        movers.append(num_nodes_moving)
        internal, external = shardmap_evaluate(shardMap, numNodes, neighborsMap)
        edgeFracs.append(float(internal)/(internal+external))
        print(istr + 'Internal edge fraction = ', float(internal)/(internal+external))
#         edge_score = score(num_nodes, neighborsMap, num_partitions, assignments, edges)
#         edge_score_values.append(edge_score)
        if return_periodicity:
            assert(len(shardMap) == numNodes)
            for node in range(numNodes):
                period = i - assignmentHistory[node, shardMap[node]] if assignmentHistory[node, shardMap[node]] != np.inf else 0
                periodicity[int(period), i] += 1
                assignmentHistory[node, shardMap[node]] = i

    return edgeFracs, movers, periodicity
