import numpy as np
from shardmap import *

def calculate_preferences(numNodes, neighborsMap, shardMap, numShards):
    requestMatrix = np.zeros((numShards, numShards), dtype=np.int32)
    moverMap = defaultdict(list)
    for node in range(numNodes):
        if len(neighborsMap[node]) > 0:
            first = shardMap[node]
            second = -1
            currentHist = {i: 0 for i in range(numShards)}
            np.random.shuffle(neighborsMap[node])
            for v in neighborsMap[node]:
                currentHist[shardMap[v]] += 1
                if currentHist[shardMap[v]] > currentHist[first]:
                    second = first
                    first = shardMap[v]
                elif (second == -1) or (currentHist[second] < currentHist[shardMap[v]] <= currentHist[first]):
                    second = shardMap[v]
                else:
                    continue
            if first != shardMap[node]:
                requestMatrix[shardMap[node], first] += 1
                improvement = currentHist[first] - currentHist[shardMap[node]]
                moverMap[(shardMap[node], first)].append((node, improvement))
            else:
                improvement = currentHist[second] - currentHist[shardMap[node]]
                moverMap[(shardMap[node], second)].append((node, improvement))
        else:
            continue
    return requestMatrix, moverMap

def version_I_relocation(numNodes, neighborsMap, shardMap, numShards, requestMatrix, moverMap, moveCounter, return_periodicity, periodicities, assignmentHistory, i):
    P = np.minimum(requestMatrix.astype(np.int32), (requestMatrix.astype(np.int32)).T)
    for j in range(numShards):
        krange = list(range(numShards))
        krange.remove(j)
        for k in krange:
            requestList = moverMap[(j,k)]
            movingNodes = [node for (node, improve) in requestList if improve > 0.0]
            np.random.shuffle(movingNodes)
            for node in movingNodes[:P[j,k]]:
                shardMap[node] = k
                moveCounter += 1
            if return_periodicity:
                for node in movingNodes[:P[j,k]]:
                    period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                    periodicities[int(period)] += 1
                    assignmentHistory[node, k] = i + 1

def version_II_relocation(numNodes, neighborsMap, shardMap, numShards, requestMatrix, moverMap, moveCounter, return_periodicity, periodicities, assignmentHistory, i):
    P = np.minimum(requestMatrix.astype(np.int32), (requestMatrix.astype(np.int32)).T)
    for j in range(numShards):
        krange = list(range(numShards))
        krange.remove(j)
        for k in krange:
            requestList = moverMap[(j,k)]
            requestList.sort(key=lambda y: (-y[1], y[0]))
            movingNodes = [node for (node, improve) in requestList if improve > 0.0]
            for node in movingNodes[:P[j,k]]:
                shardMap[node] = k
                moveCounter += 1
            if return_periodicity:
                for node in movingNodes[:P[j,k]]:
                    period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                    periodicities[int(period)] += 1
                    assignmentHistory[node, k] = i + 1          

def version_kl_relocation(numNodes, neighborsMap, shardMap, numShards, moverMap, moveCounter, c, return_periodicity, periodicities, assignmentHistory, i):
    for j in range(numShards):
        for k in range(j+1, numShards):
            forwardRequestList = moverMap[(j,k)]
            backwardRequestList = moverMap[(k,j)]
            # sort by decreasing gain
            forwardRequestList.sort(key=lambda y: (-y[1], y[0]))
            backwardRequestList.sort(key=lambda y: (-y[1], y[0]))
            # threshold movers to gains > c
            forwardRequestThresh = [(node, improvement) for (node, improvement) in forwardRequestList if improvement > c]
            backwardRequestThresh = [(node, improvement) for (node, improvement) in backwardRequestList if improvement > c]
            if len(forwardRequestThresh) == 0 or len(backwardRequestThresh) == 0:
                continue
            forwardNodes, forwardImprovements = zip(*forwardRequestThresh)
            backwardNodes, backwardImprovements = zip(*backwardRequestThresh)
            length = min(len(forwardImprovements), len(backwardImprovements))
            netImprovement = np.array(forwardImprovements[:length]) + np.array(backwardImprovements[:length])
            # finding cut off of swaps to yield positive gain
            if netImprovement[-1] > 0.0:
                index = length
            else:
                index = np.argmax(netImprovement <= 0.0)
            # make forward moves, up to index
            for node in forwardNodes[:index]:
                shardMap[node] = k
                moveCounter += 1
                if return_periodicity:
                    period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                    periodicities[int(period)] += 1
                    assignmentHistory[node, k] = i + 1
            # make backward moves, up to index
            for node in backwardNodes[:index]:
                shardMap[node] = j
                moveCounter += 1
                if return_periodicity:
                    period = i + 1 - assignmentHistory[node, j] if assignmentHistory[node, j] != np.inf else 0
                    periodicities[int(period)] += 1
                    assignmentHistory[node, j] = i + 1
        
def shp(numNodes, neighborsMap, numShards, numIterations, c = -np.inf, return_periodicity = False, initialization = None, version = 'KL'):
    print('Loading or generating random initialization...')
    if initialization is not None:
        shardMap = initialization
    else:
        shardMap = shardmap_random_init(numNodes, numShards)
    edgeFracs = []
    moveCounter = 0
    internal, external = shardmap_evaluate(shardMap, numNodes, neighborsMap)
    edgeFracs.append(float(internal)/(internal+external))
    
    if return_periodicity:
        print('Initializing structures for collecting periodicity data...')
        periodicity = np.zeros((numIterations+1, numIterations))
        assignmentHistory = np.full((numNodes, numShards), np.inf)
        for i, j in enumerate(shardMap):
            assignmentHistory[i,j] = 0
    else:
        periodicity = None
        assignmentHistory = None
        
    print('Beginning SHP, version ' + version + '...')
    for i in range(numIterations):
        istr = str(i)+ ": "
        requestMatrix, moverMap = calculate_preferences(numNodes, neighborsMap, shardMap, numShards)
        periodicities = np.zeros(numIterations+1)
        if version == 'I':
            version_I_relocation(numNodes, neighborsMap, shardMap, numShards, requestMatrix, moverMap, moveCounter, return_periodicity, periodicities, assignmentHistory, i)
        if version == 'II':
            version_II_relocation(numNodes, neighborsMap, shardMap, numShards, requestMatrix, moverMap, moveCounter, return_periodicity, periodicities, assignmentHistory, i)
        if version == 'KL':
            version_kl_relocation(numNodes, neighborsMap, shardMap, numShards, moverMap, moveCounter, c, return_periodicity, periodicities, assignmentHistory, i)
        if return_periodicity:
            periodicities[1] = numNodes - moveCounter
            for period, amount in enumerate(periodicities):
                periodicity[period, i] = amount

        internal, external = shardmap_evaluate(shardMap, numNodes, neighborsMap)
        edgeFracs.append(float(internal)/(internal+external))
        print(istr + 'Internal edge fraction = ', float(internal)/(internal+external))

    return edgeFracs, periodicity
