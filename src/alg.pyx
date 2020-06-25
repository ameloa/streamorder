import numpy as np
cimport cython

cdef int UNMAPPED = -1

def linear_deterministic_greedy(int[:,::] edges,
                                int [::] stream_order_indices,
                                int num_nodes,
                                int num_partitions,
                                int[::] partition,
                                int[::] currentHist,
                                float epsilon
                               ):
    """
    This algorithm favors a cluster if it has many neighbors of a node, but
    penalizes the cluster if it is close to capacity.

    edges: An [:,2] array of edges.
    stream_order_indices: A list of indices in which to stream over the edges.
    num_nodes: The number of nodes in the graph.
    num_partitions: How many partitions we are breaking the graph into.
    partition: The partition from a previous run. Used for restreaming.

    Returns: A new partition.
    """
    # The output partition

    if partition is None:
        partition = np.repeat(np.int32(UNMAPPED), num_nodes)

    cdef int[::] partition_sizes = np.zeros(num_partitions, dtype=np.int32)
    if currentHist is not None:
        partition_sizes = currentHist
            
    cdef int[::] partition_votes = np.zeros(num_partitions, dtype=np.int32)

    # Fine to be a little off, to stay integers
    cdef int partition_capacity = int((num_nodes / num_partitions) * (1.0 + epsilon))

    cdef int last_left = edges[0,0]
    cdef int i = 0
    cdef int left = 0
    cdef int right = 0
    cdef int arg = 0
    cdef int max_arg = 0
    cdef int max_val = 0
    cdef int val = 0
    cdef int len_edges = len(edges)
    cdef int num_nodes_moved = 0

    for i in stream_order_indices:
        left = edges[i,0]
        right = edges[i,1]

        if last_left != left:
            # We have found a new node so assign last_left to a partition
            max_arg = 0
            max_val = (partition_votes[0]) * (
                       partition_capacity - partition_sizes[0])

            for arg in range(1, num_partitions):
                val = (partition_votes[arg]) * (
                       partition_capacity - partition_sizes[arg])
                if val > max_val:
                    max_arg = arg
                    max_val = val

            if max_val == 0:
                max_arg = arg
                # No neighbors (or multiple maxed out) so "randomly" select
                # the smallest partition
                for arg in range(i % num_partitions, num_partitions):
                    if partition_sizes[arg] < partition_capacity:
                        max_arg = arg
                        max_val = 1
                        break
                if max_val == 0:
                    for arg in range(0, i % num_partitions):
                        if partition_sizes[arg] < partition_capacity:
                            max_arg = arg
                            break
            if max_arg != partition[last_left]:
                num_nodes_moved += 1
            partition_sizes[max_arg] += 1
            partition[last_left] = max_arg
            partition_votes[:] = 0
            last_left = left
        if (right < 0) or (right >= num_nodes):
            print(right, num_nodes)
        if partition[right] != UNMAPPED:
            partition_votes[partition[right]] += 1

    # Clean up the last assignment
    max_arg = 0
    max_val = 0
    for arg in range(0, num_partitions):
        if partition_sizes[arg] < partition_capacity:
            val = (partition_votes[arg]) * (
                1 - int(partition_sizes[arg]) / partition_capacity)
            if val > max_val:
                max_arg = arg
                max_val = val
    if max_arg != partition[left]:
        num_nodes_moved += 1
    partition[left] = max_arg

    return (num_nodes_moved,  np.asarray(partition))
