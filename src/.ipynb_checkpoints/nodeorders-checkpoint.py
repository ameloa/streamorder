import numpy as np
from os import path
from collections import deque

def compute_clustering_coefficients(data_directory, edge_list, num_nodes, neighborsMap):
    import_path = data_directory + edge_list + '_ccs.txt'
    if path.exists(import_path):
        cc = np.loadtxt(import_path)
        assert(len(cc) == num_nodes)
    else:
        cc = np.zeros(num_nodes)
        for node in range(num_nodes):
            if len(neighborsMap[node]) < 2:
                continue
            deg = len(neighborsMap[node])
            triangles = 0
            neighbors = neighborsMap[node]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighborsMap[node][j] in neighborsMap[neighborsMap[node][i]]:
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

def bfs_disconnected(data_directory, edge_list, num_nodes, neighborsMap, degrees):
    import_path = data_directory + edge_list + '_bfs.txt'
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
    if not path.exists(import_path):
        np.savetxt(import_path, order)
    return order, connected_components
