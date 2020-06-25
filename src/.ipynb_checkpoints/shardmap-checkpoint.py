import numpy as np
import sys
import random
from collections import defaultdict
from math import ceil

UNMAPPED = -1

def shardmap_random_init(numNodes, numShards):
    shardMap = np.full((numNodes), UNMAPPED)
    assignments = list(np.repeat(np.arange(numShards), int(ceil( numNodes / float(numShards) ))))
    random.shuffle(assignments)
    for node in range(numNodes):
        shardMap[node] = assignments.pop()
    return shardMap

def shardmap_calculate_requests(numNodes, neighborsMap, shardMap, numShards, c = 0.0):
    populationCount = {}
    for i in range(numShards):
        populationCount[i] = 0

    improvementCount = defaultdict(int)
    for i in range(numShards):
        jrange = list(range(numShards))
        jrange.remove(i)
        for j in jrange:
            improvementCount[(i, j, 1)] = 0

    moverMap = defaultdict(list)
    for node in range(numNodes):
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

    requestList = [ [x[0], x[1], x[2], improvementCount[x]] for x in improvementCount]
    requestList.sort(key=lambda y: (y[0], y[1], -y[2]))
    return (requestList, moverMap, populationCount)

def shardmap_move_leaders(moverMap, variables, shardMap, numShards, numIterations, periodicity = False, assignmentHistory = None, i = None):
    moveCounter = 0
    periodicities = np.zeros(numIterations+1)
    for pair in moverMap:
        currentMovers = moverMap[pair]
        currentMovers.sort(key= lambda y: (-y[1], random.random()))
        nodesInOrder = [t[0] for t in currentMovers]
        nodesToMove = nodesInOrder[:variables[pair]]
        for node in nodesToMove:
            shardMap[node] = pair[1]
            if periodicity:
                period = i - assignmentHistory[node, pair[1]] if assignmentHistory[node, pair[1]] != np.inf else 0
                periodicities[int(period)] += 1
                assignmentHistory[node, pair[1]] = i
            moveCounter += 1
            
    return moveCounter, periodicities

def shardmap_update(numNodes, histogramMap, thresholdMap, marginMap, shardMap):
    shardMap2 = {}
    moves = 0
    for node in range(numNodes):
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
                    newShard = maxShard
                    moves += 1
                else:
                    newShard = oldShard
            else:
                newShard = oldShard
        shardMap2[node] = newShard
    return (shardMap2, moves)

def shardmap_evaluate(shardMap, numNodes, neighborsMap):
    internal = 0
    external = 0
    for u in range(numNodes):
        for v in neighborsMap[u]:
            if (shardMap[u] == shardMap[v]):
                internal += 1
            else:
                external += 1
    return (internal, external)
