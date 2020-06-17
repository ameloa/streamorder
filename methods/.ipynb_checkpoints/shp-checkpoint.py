import numpy as np
from listshardmap import *

# @profile
def ComputeMoveGain(neighbor_shards, shard, p, currentShard):
    num_there = np.count_nonzero(neighbor_shards == shard)
    num_here = np.count_nonzero(neighbor_shards == currentShard)
    gain = (1.0-p)*num_there - (1.0-p)*num_here
    if p==0.0:
        return gain
    else:
        return p*gain

# @profile
def ComputeFanout(nodes, neighborsMap, shardMap):
    total = 0.0
    total_degree = 0
    for node in nodes:
        # neighbor_shards = np.array([shardMap[neighbor] for neighbor in neighborsMap[node]])
        # total_degree += len(neighbor_shards)
        # total += float(total_degree) - 0.5 * np.count_nonzero(neighbor_shards == shardMap[node])
        for neighbor in neighborsMap[node]:
            total_degree += 1
            neighborShard = shardMap[neighbor]
            total = total + 0.5 if neighborShard == shardMap[node] else total + 1.0
    return total/float((total_degree//2))

# @profile
def CalculatePreferences(nodes, neighborsMap, shardMap, numOfMachines):
    S = np.zeros((numOfMachines, numOfMachines), dtype=np.int32)
    moverMap = defaultdict(list)
    for node in nodes:
        if len(neighborsMap[node]) > 0:
            first = shardMap[node]
            second = -1
            currentHist = {i: 0 for i in range(numOfMachines)}
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
                S[shardMap[node], first] += 1
                improvement = currentHist[first] - currentHist[shardMap[node]]
                moverMap[(shardMap[node], first)].append((node, improvement))
            else:
                improvement = currentHist[second] - currentHist[shardMap[node]]
                moverMap[(shardMap[node], second)].append((node, improvement))
        else:
            continue
    return S, moverMap
        
# @profile
def SocialHash(nodes, neighborsMap, NUMBER_OF_SHARDS, PROP_ITERATIONS, p=0.0, PERIODICITY=False, INITIAL_STREAM=None, TYPE=2, SORT=1, THRESHOLD=False, c=-np.inf):
    shardMap = shardmap_random_init(nodes, NUMBER_OF_SHARDS) if INITIAL_STREAM.all() == None else INITIAL_STREAM
    edgeFracs, movers, requests = [], [], []
    if PERIODICITY:
        periodicity = np.zeros((PROP_ITERATIONS+1, PROP_ITERATIONS))
        # Initialize array which stores history of nodes assignment throughout trial
        assignmentHistory = np.full((len(nodes), NUMBER_OF_SHARDS), np.inf)
        for i, j in enumerate(shardMap):
            assignmentHistory[i,j] = 0
    else:
        periodicity = None

    # (internal, external) = shardmap_evaluate(shardMap, nodes, neighborsMap)
    # edgeFracs.append(float(internal)/(internal+external))
    # p_fanout.append(ComputeFanout(nodes, neighborsMap, NUMBER_OF_SHARDS, shardMap))

    for i in range(PROP_ITERATIONS):
        numMoving = 0
        S, moverMap = CalculatePreferences(nodes, neighborsMap, shardMap, NUMBER_OF_SHARDS)
        if PERIODICITY:
            periodicities = np.zeros(PROP_ITERATIONS+1)
        if TYPE == 1:
            P = np.divide(np.minimum(S, S.T), S.astype(np.float64))
            for j in range(NUMBER_OF_SHARDS):
                krange = list(range(NUMBER_OF_SHARDS))
                krange.remove(j)
                for k in krange:
                    requestList = moverMap[(j,k)]
                    movingNodes = [node for (node, improve) in requestList if improve > 0.0]
                    for node in movingNodes:
                        if np.random.uniform() < P[j,k]:
                            numMoving += 1
                            shardMap[node] = k
                        if PERIODICITY:
                            period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                            periodicities[int(period)] += 1
                            assignmentHistory[node, k] = i + 1
        if TYPE == 2 and SORT == 1:
            P = np.minimum(S.astype(np.int32), (S.astype(np.int32)).T)
            for j in range(NUMBER_OF_SHARDS):
                krange = list(range(NUMBER_OF_SHARDS))
                krange.remove(j)
                for k in krange:
                    requestList = moverMap[(j,k)]
                    movingNodes = [node for (node, improve) in requestList if improve > 0.0]
                    np.random.shuffle(movingNodes)
                    for node in movingNodes[:P[j,k]]:
                        shardMap[node] = k
                        numMoving += 1
                    if PERIODICITY:
                        for node in movingNodes[:P[j,k]]:
                            period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                            periodicities[int(period)] += 1
                            assignmentHistory[node, k] = i + 1
        if TYPE == 2 and SORT == 2:
            P = np.minimum(S.astype(np.int32), (S.astype(np.int32)).T)
            for j in range(NUMBER_OF_SHARDS):
                krange = list(range(NUMBER_OF_SHARDS))
                krange.remove(j)
                for k in krange:
                    requestList = moverMap[(j,k)]
                    requestList.sort(key=lambda y: (-y[1], y[0]))
                    movingNodes = [node for (node, improve) in requestList if improve > 0.0]
                    for node in movingNodes[:P[j,k]]:
                        shardMap[node] = k
                        numMoving += 1
                    if PERIODICITY:
                        for node in movingNodes[:P[j,k]]:
                            period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                            periodicities[int(period)] += 1
                            assignmentHistory[node, k] = i + 1
        if TYPE == 2 and SORT == 3:
            for j in range(NUMBER_OF_SHARDS):
                for k in range(j+1, NUMBER_OF_SHARDS):
                    forwardRequestList = moverMap[(j,k)]
                    backwardRequestList = moverMap[(k,j)]
                    forwardRequestList.sort(key=lambda y: (-y[1], y[0]))
                    backwardRequestList.sort(key=lambda y: (-y[1], y[0]))
                    if THRESHOLD:
                        forwardRequestThresh = [(node, improvement) for (node, improvement) in forwardRequestList if improvement > c]
                        backwardRequestThresh = [(node, improvement) for (node, improvement) in backwardRequestList if improvement > c]
                    else:
                        forwardRequestThresh = forwardRequestList
                        backwardRequestThresh = backwardRequestList
                    if len(forwardRequestThresh) == 0 or len(backwardRequestThresh) == 0:
                        continue
                    forwardNodes, forwardImprovements = zip(*forwardRequestThresh)
                    backwardNodes, backwardImprovements = zip(*backwardRequestThresh)
                    length = min(len(forwardImprovements), len(backwardImprovements))
                    netImprovement = np.array(forwardImprovements[:length]) + np.array(backwardImprovements[:length])
                    if netImprovement[-1] > 0.0:
                        index = length
                    else:
                        index = np.argmax(netImprovement <= 0.0)
                    for node in forwardNodes[:index]:
                        shardMap[node] = k
                        numMoving += 1
                        if PERIODICITY:
                            period = i + 1 - assignmentHistory[node, k] if assignmentHistory[node, k] != np.inf else 0
                            periodicities[int(period)] += 1
                            assignmentHistory[node, k] = i + 1
                    for node in backwardNodes[:index]:
                        shardMap[node] = j
                        numMoving += 1
                        if PERIODICITY:
                            period = i + 1 - assignmentHistory[node, j] if assignmentHistory[node, j] != np.inf else 0
                            periodicities[int(period)] += 1
                            assignmentHistory[node, j] = i + 1
        if PERIODICITY:
            periodicities[1] = len(nodes) - numMoving
            for period, amount in enumerate(periodicities):
                periodicity[period, i] = amount

        (internal, external) = shardmap_evaluate(shardMap, nodes, neighborsMap)
        edgeFracs.append(float(internal)/(internal+external))
        movers.append(numMoving)

    return edgeFracs, movers, periodicity
