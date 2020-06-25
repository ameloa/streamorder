import sys
from copy import deepcopy
from collections import defaultdict
from lpbuilder import * #solve_lp
# import shardmap #shardmap_random_init, shardmap_calculate_requests, shardmap_update, shardmap_evaluate
from shardmap import * #shardmap_random_init, shardmap_calculate_requests, shardmap_update, shardmap_evaluate
# from utils import * #read_nx_edgelist, read_edgelist, plot_edgefracs
from shp import *


'''
    Implementation of Balanced Label Propagation.
    For implementation details, see the original paper by Ugander & Backstrom:
    https://stanford.edu/~jugander/papers/wsdm13-blp.pdf
'''
def blp(numNodes, neighborsMap, numShards, numIterations, epsilon, c = 0.0, return_periodicity = False, initialization = None):
    print('Loading or generating random initialization...')
    if initialization is not None:
        initialAssignment = initialization
    else:
        initialAssignment = shardmap_random_init(numNodes, numShards)
    shardMap = deepcopy(initialAssignment)
    assert((initialAssignment == shardMap).all())

    edgeFracs = []
    internal, external =  shardmap_evaluate(shardMap, numNodes, neighborsMap)
    edgeFracs.append(float(internal)/(internal+external))
    
    movers = []
    if return_periodicity:
        print('Initializing structures for collecting periodicity data...')
        periodicity = np.zeros((numIterations+1, numIterations))
        assignmentHistory = np.full((numNodes, numShards), np.inf)
        for i, j in enumerate(initialAssignment):
            assignmentHistory[i,j] = 0
    else:
        periodicity = None
    
    print('Beginning BLP...')
    for i in range(numIterations):
        istr = str(i)+ ": "
        requestList, moverMap, populationCount = shardmap_calculate_requests(numNodes, neighborsMap, shardMap, numShards, c)

        maxPop = max(populationCount.values())
        minPop = min(populationCount.values())
        relRange = (maxPop - minPop)*float(len(populationCount)) / sum(populationCount.values())

        if len(requestList) == 0:
            print(istr + 'No move requests. Done.')
            while len(edgeFracs) <= numIterations:
                edgeFracs.append(edgeFracs[-1])
                movers.append(0)
            break

        variables = solve_lp(numNodes, numShards, requestList, populationCount, epsilon)

        if return_periodicity:
            moveCounter, periodicities = shardmap_move_leaders(moverMap, variables, shardMap, numShards, numIterations, return_periodicity, assignmentHistory, i + 1)
        else:
            moveCounter, _ = shardmap_move_leaders(moverMap, variables, shardMap, numShards, numIterations)

        if return_periodicity:
            periodicities[1] = numNodes - moveCounter
            for period, amount in enumerate(periodicities):
                periodicity[period, i] = amount
        
        if (moveCounter > 0):
            movers.append(moveCounter)
            internal, external = shardmap_evaluate(shardMap, numNodes, neighborsMap)
            edgeFracs.append(float(internal)/(internal+external))
            print(istr + 'Internal edge fraction = ', float(internal)/(internal+external))
            pass
        else:
            print(istr + 'No movers. Done.')
            while len(edgeFracs) <= numIterations:
                edgeFracs.append(edgeFracs[-1])
                movers.append(0)
            break
    assert(not (initialAssignment == shardMap).all())
    return initialAssignment, edgeFracs, movers, periodicity
