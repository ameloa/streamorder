import numpy as np
import os
import sys
sys.stdout.flush()

# NODES MUST BE NUMBERED 0 - NUM_NODES-1 IN EDGES ARRAY. Generate a map before such that idMap[id] = node.

def row_generator(FILENAME):
    """This will generate all the edges in the graph."""
#     with gzip.open(WIKIVOTE_FILENAME, 'rt') as f:
    with open(FILENAME, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                (left_node, right_node) = line.split('\t')
                yield(int(left_node), int(right_node))

def to_undirected(edge_iterable, num_edges, shuffle=True):
    """Takes an iterable of edges and produces the list of edges for the undirected graph.

    > to_undirected([[0,1],[1,2],[2,10]], 3, 11)
    array([[ 0,  1],
       [ 1,  0],
       [ 1,  2],
       [ 2,  1],
       [ 2, 10],
       [10,  2]])
    """
    # need int64 to do gross bithacks
    as_array = np.zeros((num_edges, 2), dtype=np.int64)
    for (i, (n_0, n_1)) in enumerate(edge_iterable):
            as_array[i,0] = n_0
            as_array[i,1] = n_1

    # must map node values to be between 0 - num_nodes
    nodes, array = np.unique(as_array, return_inverse=True)
    as_array = array.reshape(num_edges, 2)
    num_nodes = len(nodes)

    # The graph is directed, but we want to make it undirected,
    # which means we will duplicate some rows.
    left_nodes = as_array[:,0]
    right_nodes = as_array[:,1]
    if shuffle:
        the_shuffle = np.arange(num_nodes)
        np.random.shuffle(the_shuffle)
        left_nodes = the_shuffle.take(left_nodes)
        right_nodes = the_shuffle.take(right_nodes)

    # numpy.unique will not unique whole rows, so this little bit-hacking
    # is a quick way to get unique rows after making a flipped copy of
    # each edge.
    max_bits = int(np.ceil(np.log2(num_nodes + 1)))

    encoded_edges_forward = np.left_shift(left_nodes, max_bits) | right_nodes

    # Flip the columns and do it again:
    encoded_edges_reverse = np.left_shift(right_nodes, max_bits) | left_nodes

    unique_encoded_edges = np.unique(np.hstack((encoded_edges_forward, encoded_edges_reverse)))

    left_node_decoded = np.right_shift(unique_encoded_edges, max_bits)

    # Mask out the high order bits
    right_node_decoded = (2 ** (max_bits) - 1) & unique_encoded_edges

    undirected_edges = np.vstack((left_node_decoded, right_node_decoded)).T.astype(np.int32)

    # for sorting use later
    neighborsMap = [[] for node in range(num_nodes)]
    node_indices = [[] for node in range(num_nodes)]
    for i in range(undirected_edges.shape[0]):
        neighborsMap[undirected_edges[i,0]].append(undirected_edges[i,1])
        node_indices[undirected_edges[i,0]].append(i)

    # ascontiguousarray so that it's c-contiguous for cython code
    return np.array(neighborsMap), num_nodes, np.array(node_indices), np.ascontiguousarray(undirected_edges)

def get_clean_data(DATA_DIRECTORY, EDGE_LIST, NUM_EDGES, shuffle=True):
    if os.path.exists(DATA_DIRECTORY + EDGE_LIST + '_edges.txt'):
        edges = np.loadtxt(DATA_DIRECTORY + EDGE_LIST + '_edges.txt')
        num_nodes = len(np.unique(edges))
        neighborsMap = [[] for node in range(num_nodes)]
        node_indices = [[] for node in range(num_nodes)]
        for i in range(edges.shape[0]):
            neighborsMap[edges[i,0]].append(edges[i,1])
            node_indices[edges[i,0]].append(i)
    else:
        neighborsMap, num_nodes, node_indices, edges = to_undirected(row_generator(DATA_DIRECTORY + EDGE_LIST + '.txt'), NUM_EDGES, shuffle=shuffle)
    return neighborsMap, num_nodes, node_indices, edges
