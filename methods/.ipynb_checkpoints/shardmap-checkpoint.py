import numpy as np
import sys
import random
from collections import defaultdict
from math import ceil

UNMAPPED = -1

def shardmap_random_init(nodes, NUMBER_OF_SHARDS):
    shardMap = [UNMAPPED for node in nodes]
    assignments = list(np.repeat(np.arange(NUMBER_OF_SHARDS), int(ceil( len(nodes) / float(NUMBER_OF_SHARDS) ))))
    random.shuffle(assignments)
    for node in nodes:
        shardMap[node] = assignments.pop()
    return shardMap

def shardmap_load(nodes, initial_filename):
    ''' This first loads data to a dict
    and then sees which have been used. A bit
    wonky, but it's important for the code that
    shardMap only contain nodes that exist.
    '''
    shardMap = {}
    shardMap_from_file = {}
    f = open(initial_filename, 'r')
    for line in f.xreadlines():
        vec = line.rstrip().split(" ")
        shardMap_from_file[int(vec[0])] = int(vec[1])
    for node in nodes:
        shardMap[node] = shardMap_from_file[node]
    return shardMap

# @profile
def shardmap_calculate_requests(nodes, neighborsMap, shardMap, numOfMachines, c = 0.0):
    # histogramMap = {}
    # defaultdict is not a good option here in low-node cases where not all
    # machines have a population. (super edge-case, but yes).
    populationCount = {}
    for i in range(numOfMachines):
        populationCount[i] = 0

    improvementCount = defaultdict(int)
    for i in range(numOfMachines):
        jrange = list(range(numOfMachines))
        jrange.remove(i)
        for j in jrange:
            improvementCount[(i, j, 1)] = 0

    moverMap = defaultdict(list)
    for node in nodes:
        populationCount[shardMap[node]] += 1
        currentHist = defaultdict(int)

        # only interested in moving if have neighbors
        if len(neighborsMap[node]) > 0:
            for v in neighborsMap[node]:
                currentHist[shardMap[v]] += 1
            maxShard = max(currentHist, key=currentHist.get)
            improvement = currentHist[maxShard] - currentHist[shardMap[node]]
            if (improvement > c) and (maxShard != shardMap[node]):
                improvementTuple = (
                    shardMap[node],
                    maxShard,
                    improvement
                )
                improvementCount[improvementTuple] += 1
                # for each (i,j) pair, moverMap contains a list of (user, improvement) tuples,
                # which decides who gets to move at the end
                moverMap[(shardMap[node], maxShard)].append((node, improvement))

        # histogramMap[node] = currentHist
    requestList = [ [x[0], x[1], x[2], improvementCount[x]] for x in improvementCount]
    # random.shuffle(requestList)
    requestList.sort(key=lambda y: (y[0], y[1], -y[2]))
    return (requestList, moverMap, populationCount)

# @profile
def shardmap_move_leaders(moverMap, variables, shardMap, numOfMachines, numIterations, PERIODICITY = False, assignmentHistory = None, i = None):
    movers = np.zeros((numOfMachines,numOfMachines))
    moveCounter = 0
    periodicities = np.zeros(numIterations+1)
    for pair in moverMap:
        currentMovers = moverMap[pair]
        currentMovers.sort(key= lambda y: (-y[1], random.random()))
        nodesInOrder = [t[0] for t in currentMovers]
        nodesToMove = nodesInOrder[:variables[pair]]
        for node in nodesToMove:
            movers[shardMap[node], pair[1]] += 1
            shardMap[node] = pair[1]
            if PERIODICITY:
                periodicity = i - assignmentHistory[node, pair[1]] if assignmentHistory[node, pair[1]] != np.inf else 0
                periodicities[int(periodicity)] += 1
                assignmentHistory[node, pair[1]] = i
            moveCounter += 1
            
    return moveCounter, movers, periodicities

def shardmap_update(nodes, histogramMap, thresholdMap, marginMap, shardMap):
    # this function is the hadoop logic, deprecated
    shardMap2 = {}
    moves = 0
    for node in nodes:
        currentHist = histogramMap[node]
        maxShard = max(currentHist, key=currentHist.get)
        oldShard = shardMap[node]
        improvement = currentHist[maxShard] - currentHist[oldShard]
        if (improvement <= 0):
            newShard = oldShard
        else:
            if (improvement > thresholdMap[(oldShard, maxShard)]):
                newShard = maxShard
                moves += 1
            elif (improvement == thresholdMap[(oldShard, maxShard)]):
                if (random.random() < marginMap[(oldShard, maxShard)]):
                    # pctOnMargin is the fraction to be moved
                    newShard = maxShard
                    moves += 1
                else:
                    newShard = oldShard
            else:
                newShard = oldShard
        shardMap2[node] = newShard
    return (shardMap2, moves)

# @profile
def shardmap_evaluate(shardMap, nodes, neighborsMap):
    internal = 0
    external = 0
    for u in nodes:
        for v in neighborsMap[u]:
            if (shardMap[u] == shardMap[v]):
                internal += 1
            else:
                external += 1
    return (internal, external)
