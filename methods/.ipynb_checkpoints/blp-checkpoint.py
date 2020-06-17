import networkx as nx
import sys
import random
import time
from copy import deepcopy
from collections import defaultdict
from lpbuilder import * #solve_lp
# import shardmap #shardmap_random_init, shardmap_calculate_requests, shardmap_update, shardmap_evaluate
import shardmap #shardmap_random_init, shardmap_calculate_requests, shardmap_update, shardmap_evaluate
# from utils import * #read_nx_edgelist, read_edgelist, plot_edgefracs
from shp import *

def blp(nodes, neighborsMap, NUMBER_OF_SHARDS, PROP_ITERATIONS, absoluteFraction, PERIODICITY, c, initializations = None):

#     sys.stderr.write('Random initialization...\n')
    if initializations is not None:
        initialassignment = initializations
    else:
        initialassignment = shardmap.shardmap_random_init(nodes, NUMBER_OF_SHARDS)
    shardMap = deepcopy(initialassignment)

    edgeFracs = []
    (internal, external) =  shardmap.shardmap_evaluate(shardMap, nodes, neighborsMap)
    edgeFracs.append(float(internal)/(internal+external))
    
    movers = []
    moverMatrices = np.zeros((PROP_ITERATIONS, NUMBER_OF_SHARDS, NUMBER_OF_SHARDS))
    if PERIODICITY:
        periodicity = np.zeros((PROP_ITERATIONS+1, PROP_ITERATIONS))
        # Initialize array which stores history of nodes assignment throughout trial
        assignmentHistory = np.full((len(nodes), NUMBER_OF_SHARDS), np.inf)
        for i, j in enumerate(initialassignment):
            assignmentHistory[i,j] = 0
    else:
        periodicity = None
    
    for i in range(PROP_ITERATIONS):
        istr = str(i) + ": "
#         sys.stderr.write(istr + 'Building move requests...\n')

        (requestList, moverMap, populationCount) = shardmap.shardmap_calculate_requests(nodes, neighborsMap, shardMap, NUMBER_OF_SHARDS, c)

        maxPop = max(populationCount.values())
        minPop = min(populationCount.values())
        relRange = (maxPop - minPop)*float(len(populationCount)) / sum(populationCount.values())
#         sys.stderr.write(istr + 'Max-Min populations: %d-%d, range=  \n' % (minPop, maxPop, relRange) )

        if len(requestList) == 0:
#             sys.stderr.write(istr + 'No moves requested. Finished.\n')
            while len(edgeFracs) <= PROP_ITERATIONS:
                edgeFracs.append(edgeFracs[-1])
                movers.append(0)
            break
#         sys.stderr.write(istr + 'Running LP...\n')

        variables = solve_lp(requestList, populationCount, absoluteFraction)

#         sys.stderr.write(istr + 'Finished LP, moving leaders...\n')
        if PERIODICITY:
            moveCounter, moverMatrix, periodicities = shardmap.shardmap_move_leaders(moverMap, variables, shardMap, NUMBER_OF_SHARDS, PROP_ITERATIONS, PERIODICITY, assignmentHistory, i + 1)
        else:
            moveCounter, moverMatrix, _ = shardmap.shardmap_move_leaders(moverMap, variables, shardMap, NUMBER_OF_SHARDS, PROP_ITERATIONS)
        if PERIODICITY:
            periodicities[1] = len(nodes) - moveCounter
            for period, amount in enumerate(periodicities):
                periodicity[period, i] = amount
        
        if (moveCounter > 0):
            movers.append(moveCounter)
            moverMatrices[i,:,:] = moverMatrix
            (internal, external) = shardmap.shardmap_evaluate(shardMap, nodes, neighborsMap)
            edgeFracs.append(float(internal)/(internal+external))
            pass
#             sys.stderr.write(istr + 'Moved ' + str(moveCounter) + ' nodes.\n')
        else:
#             sys.stderr.write(istr + 'No moves possible. Finished.\n')
            while len(edgeFracs) <= PROP_ITERATIONS:
                edgeFracs.append(edgeFracs[-1])
                movers.append(0)
            break

#         (internal, external) = shardmap.shardmap_evaluate(shardMap, nodes, neighborsMap)
#         if edgeFracs[-1] != float(internal)/(internal+external):
#             edgeFracs.append(float(internal)/(internal+external))
#             # p_fanout.append(ComputeFanout(nodes, neighborsMap, NUMBER_OF_SHARDS, shardMap))

#             # sys.stderr.write(istr + (' Internal edge frac: %.4f' % edgeFracs[-1]) + '\n')
#             # sys.stderr.write('--------------------------------------------------\n')
#         else:
#             # sys.stderr.write(istr + 'No improvement on edge fraction. Finished.\n')
#             while len(edgeFracs) <= PROP_ITERATIONS & len(movers) < PROP_ITERATIONS:
#                 edgeFracs.append(edgeFracs[-1])
#                 movers.append(0)
#             break

    # # What's this for?
    # def padded_binary_string(num, padLength=10):
    #     bin1 = str(bin(num))[2:][::-1]
    #     return (bin1 + '0'*(padLength-len(bin1)))
    #
    # shardStringMap = defaultdict(str)
    # for u in nodes:
    #     shardStringMap[u] = padded_binary_string(shardMap[u], 7)

    # f_out = open('./results/' + shard_output_filename, 'w+')
    # f_out.write("Number of nodes: " + str(len(nodes)) + "\n")
    # f_out.write("Number of edges: " + str(float(internal+external)/2.0) + "\n")
    # f_out.write("Fraction of internal edges: " + str(edgeFracs[-1]) + "\n")
    # f_out.write('--------------------------------------------------\n')
    # for node in nodes:
    #     f_out.write("User " + str(node) + ": " + str(shardMap[node]) + " (" + str(populationCount[shardMap[node]])+ ")\n")
    # f_out.close()
    # # sys.stderr.write('Partition results written to file.\n')

    return initialassignment, edgeFracs, movers, moverMatrices, periodicity
