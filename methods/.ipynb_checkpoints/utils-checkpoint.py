import sys
sys.path.append('..')
import os.path
from os import path
import time
import numpy as np
from collections import deque
from operator import itemgetter 
from alg import linear_deterministic_greedy
from socialhash import *

# from tqdm import tqdm
# @profile
def score(num_nodes, neighborsMap, num_partitions, assignment, edges):
    """Compute the score given an assignment of vertices.

    N nodes are assigned to clusters 0 to K-1.

    assignment: Vector where N[i] is the cluster node i is assigned to.
    edges: The edges in the graph, assumed to have one in each direction

    Returns: (total wasted bin space, ratio of edges cut)
    """
    # fanout = ComputeFanout(np.arange(num_nodes), neighborsMap, num_partitions, assignment)
    # balance = np.bincount(assignment) / len(assignment)
    # waste = (np.max(balance) - balance).sum()

    left_edge_assignment = assignment.take(edges[:,0])
    right_edge_assignment = assignment.take(edges[:,1])
    match = (left_edge_assignment == right_edge_assignment).sum()
    int_ratio = float(match) / float(len(edges))
    # return (int_ratio, fanout)
    return int_ratio

# @profile
def get_edge_indices(order, node_indices):
# returns indices of correct sort order in edges
    sorted_edge_indices = []
    for node in order:
        sorted_edge_indices.extend(node_indices[int(node)])
    return np.asarray(sorted_edge_indices, dtype=np.int32)

# @profile
def compute_gains(num_nodes, adjacency_list, normed=False, assignments=None):
    gains = np.zeros(num_nodes)
    for node in range(num_nodes):
        neighbor_shards = [assignments[v] for v in adjacency_list[node]]
        unique, counts = np.unique(np.array(neighbor_shards), return_counts = True)
        indices = np.nonzero(unique != assignments[node])
        best = max(counts[indices]) if len(counts[indices]) != 0 else 0
        my_neighbor_count = np.sum(neighbor_shards == assignments[node])
        gain = best - my_neighbor_count
        gains[node] = gain
#     if normed:
#         ambivalence[i] = ambiv / (0.46*(degree**0.50)+0.28) # web-Stanford
#         ambivalence[i] = ambiv / (0.21*(degree**0.60)+0.73) # web-NotreDame
#         ambivalence[i] = ambiv / (0.33*(degree**0.55)+0.54) # email-Enron
#         ambivalence[i] = ambiv / (0.39*(degree**0.52)+0.51) # wiki-Vote
    return gains

def compute_clustering_coefficients(edge_list, num_nodes, adjacency_list, node_hash=None):
    import_path = './data/' + edge_list + '_ccs.txt'
    if path.exists(import_path):
        cc = np.loadtxt(import_path)
        assert(len(cc) == num_nodes)
    else:
        cc = np.zeros(num_nodes)
        for node in range(num_nodes):
            if len(adjacency_list[node]) < 2:
                continue
            deg = len(adjacency_list[node])
            triangles = 0
            neighbors = adjacency_list[node]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if adjacency_list[node][j] in adjacency_list[adjacency_list[node][i]]:
                        triangles += 1
            cc[node] = 2.0*float(triangles)/float(deg*(deg-1))
            np.savetxt(import_path, cc)
    return cc

def bfs_order(start, neighborsMap, visited, hashMap):
    q = deque()
    q.append(start)
    visited[start] = True
    bfs = []
    count = 0
    while len(q) > 0:
        curr_node = q.popleft()
        bfs.append(curr_node)
        count += 1
        inds = np.argsort(hashMap[neighborsMap[curr_node]])
        for i in inds:
            neighbor = neighborsMap[curr_node][i]
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)
    return bfs, count

def bfs_disconnected(num_nodes, neighborsMap, degrees, hashMap=None):
    if hashMap is None:
        hashMap = np.arange(num_nodes)
    nodes = np.argsort(-degrees)
    visited = np.full(num_nodes, False)
    order = []
    connected_components = []
    for i in range(len(nodes)):
        if not visited[nodes[i]]:
            curr_order, num_reached = bfs_order(nodes[i], neighborsMap, visited, hashMap)
            order.extend(curr_order)
            connected_components.append(num_reached)
    assert(len(order) == num_nodes)
    assert(sum(connected_components) == num_nodes)
    return order, connected_components
            
# @profile
def run_restreaming_greedy(i, edge_list, edges, neighborsMap, node_indices, num_nodes, hashMap, num_partitions=16, num_iterations=10, eps=0.0, degree_first_step=False, sort_type_value=1, periodicity=False, gain_histogram=False, amb_histogram=False, amb_deg_histogram=False, thresholding=False, c=-np.inf, assignments=None):
    edge_score_values = []
    nodes_moving = np.zeros(num_iterations)
    degree_file = './data/' + edge_list + '_degrees.txt'
    if path.exists(degree_file):
        degrees = np.loadtxt(degree_file)
    else:
        degrees = np.array([len(neighborsMap[node]) for node in range(num_nodes)])
        np.savetxt(degree_file, degrees)
    
    if periodicity:
        periodicity_array = np.zeros((num_iterations, num_iterations))
        assignmentHistory = np.full((num_nodes, num_partitions), np.inf)
    else:
        periodicity_array = None

    if amb_histogram:
        ambivalences = np.empty((num_iterations, num_nodes))
    elif amb_deg_histogram:
        ambivalences = np.empty((num_iterations+1, num_nodes))
        ambivalences[-1, :] = degrees
    else:
        ambivalences=None
        
    if gain_histogram:
        gains = np.empty((num_iterations, num_nodes))
    else:
        gains=None
        
    if sort_type_value == 1 or (assignments is None):
        order = hashMap
        indices = get_edge_indices(order, node_indices)
    if sort_type_value == 2:
        order = np.lexsort((hashMap, -degrees))
        indices = get_edge_indices(order, node_indices)
    if sort_type_value == 3:
        ccs = compute_clustering_coefficients(edge_list, num_nodes, neighborsMap)
        order = np.lexsort((hashMap, -ccs))
        indices = get_edge_indices(order, node_indices)
    if sort_type_value == 4:
        ccs = compute_clustering_coefficients(edge_list, num_nodes, neighborsMap)
        order = np.lexsort((hashMap, ccs))
        indices = get_edge_indices(order, node_indices)
    if sort_type_value == 5:
        bfs_file = './data/' + edge_list + '_bfs.txt'
        order = np.loadtxt(bfs_file)[i] if path.exists(bfs_file) else bfs_disconnected(num_nodes, neighborsMap, degrees, hashMap)
        if not path.exists(bfs_file):
            np.savetxt(bfs_file, order)
        indices = get_edge_indices(order, node_indices)
    for i in range(num_iterations):
        if (degree_first_step) and (i == 0):
            stream_indices = get_edge_indices(np.lexsort((hashMap,-degrees)), node_indices)
        elif sort_type_value == 6 and (assignments is not None):
            gains = compute_gains(num_nodes, neighborsMap, normed=False, assignments=assignments)
            ambivalence = -np.abs(np.array(gains))
            if thresholding:
                movingNodes = np.nonzero(gains>c)[0]
                stayingNodes = np.nonzero(gains<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], ambivalence[movingNodes]))], node_indices)
                currentHist = np.zeros(num_partitions, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[assignments[node]] += 1
            else:
                stream_indices = get_edge_indices(np.lexsort((hashMap, ambivalence)), node_indices)
        elif sort_type_value == 7 and (assignments is not None):
            gains = compute_gains(num_nodes, neighborsMap, normed=True, assignments=assignments)
            ambivalence = -np.abs(np.array(gains))
            if thresholding:
                movingNodes = np.nonzero(gains>c)[0]
                stayingNodes = np.nonzero(gains<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], -degrees[movingNodes], ambivalence[movingNodes]))], node_indices)
                currentHist = np.zeros(num_partitions, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[assignments[node]] += 1
            else:
                stream_indices = get_edge_indices(np.lexsort((hashMap, -degrees, ambivalence)), node_indices)
        elif sort_type_value == 8 and (assignments is not None):
            gain = compute_gains(num_nodes, neighborsMap, normed=True, assignments=assignments)
            gain[gain < 0.0] = 0.0
            if thresholding:
                movingNodes = np.nonzero(gain>c)[0]
                stayingNodes = np.nonzero(gain<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], -gain[movingNodes]))], node_indices)
                currentHist = np.zeros(num_partitions, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[assignments[node]] += 1
            else:
                stream_indices = get_edge_indices(np.lexsort((hashMap,-gain)), node_indices)
        else:
            if thresholding and (assignments is not None):
                gains = compute_gains(num_nodes, neighborsMap, normed=True, assignments=assignments)
                movingNodes = np.nonzero(gains>c)[0]
                stayingNodes = np.nonzero(gains<=c)[0]
                stream_indices = get_edge_indices(movingNodes[np.lexsort((hashMap[movingNodes], -gains[movingNodes]))], node_indices)
                currentHist = np.zeros(num_partitions, dtype=np.int32)
                for node in stayingNodes:
                    currentHist[assignments[node]] += 1
            else:
                stream_indices = indices
                
        if (gain_histogram or amb_histogram or amb_deg_histogram) and (assignments is not None):
            gain = compute_gains(num_nodes, neighborsMap, normed=True, assignments=assignments)
            if amb_histogram or amb_deg_histogram:
                ambivalences[i,:] = -np.abs(gain)
            if gain_histogram:
                gain[gain < 0] = 0
                gains[i,:] = gain

        if thresholding and (assignments is not None):
            num_nodes_moving, assignments = linear_deterministic_greedy(edges, stream_indices, num_nodes, num_partitions, assignments, currentHist, eps)
        else:
            num_nodes_moving, assignments = linear_deterministic_greedy(edges, stream_indices, num_nodes, num_partitions, assignments, None, eps)
        nodes_moving[i] = num_nodes_moving
        edge_score = score(num_nodes, neighborsMap, num_partitions, assignments, edges)
        edge_score_values.append(edge_score)
        if periodicity:
            assert(len(assignments) == num_nodes)
            for node in range(num_nodes):
                period = i - assignmentHistory[node, assignments[node]] if assignmentHistory[node, assignments[node]] != np.inf else 0
                periodicity_array[int(period), i] += 1
                assignmentHistory[node, assignments[node]] = i

    return edge_score_values, nodes_moving, periodicity_array, ambivalences, gains
