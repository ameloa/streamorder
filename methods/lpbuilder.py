import sys
import cvxpy as cvx
import numpy as np
from math import floor, ceil

# requestList: sorted list of lists. each list corresponds to a move request. (i, j, improvement, # of identical requests). Sorted in order of i, then j, then descending improvement.
# moverMap: dictionary. key = (i,j). value = list of (node, improvement) tuples.
# populationCount: dictionary. key = shard. value = population size.
# @profile
def solve_lp(rL, populationCount, absoluteFraction):
    # Problem data.
    numOfMachines = len(populationCount)
    numNodes = sum(populationCount.values())

    ones_vector = np.ones(numOfMachines)
    zeros_vector = np.zeros(numOfMachines)
    S, T, V = np.ones(numOfMachines), np.ones(numOfMachines), np.ones(numOfMachines)
    for i in range(numOfMachines):
        S[i] = floor((1.0 - absoluteFraction) * (float(numNodes)/float(numOfMachines)))
        T[i] = ceil((1.0 + absoluteFraction) * (float(numNodes)/float(numOfMachines)))
        V[i] = populationCount[i]

    # Construct the problem.
    Z = cvx.Variable((numOfMachines,numOfMachines)) # Objective
    X = cvx.Variable((numOfMachines,numOfMachines)) # Number of nodes to move

    messyConstraint = []
    ix = 0 # index of move request
    for i in range(numOfMachines):
        jrange = list(range(numOfMachines))
        jrange.remove(i)
        for j in jrange:
            a = {}
            b = {}
            Sum = {}
            b[0] = 0
            Sum[-1] = 0
            if (ix > len(rL) - 1):
                sys.stderr.write("-- Finished early, last move request: " + str(i) + "," + str(j) + "\n")
                break
            kix = 0
            # Add recursive constraint
            while ((int(rL[ix][0]) == i) and (int(rL[ix][1]) == j)):
                a[kix] = int(rL[ix][2])
                if (kix > 0):
                    b[kix] = b[kix - 1] + Sum[kix - 1] * (a[kix - 1] - a[kix])
                messyConstraint.append(a[kix]*X[i,j] - Z[i,j] >= -b[kix])
                Sum[kix] = Sum[kix - 1] + int(rL[ix][3])
                ix += 1
                kix += 1
                if (ix > len(rL) - 1):
                    break
            messyConstraint.append(X[i,j] <= Sum[kix - 1])

    # Build the LP
    objective = cvx.Maximize(cvx.sum(Z))
    constraints = [cvx.diag(X) == zeros_vector, # No nodes should move from a shard to itself
                    cvx.diag(Z) == zeros_vector, # Objective should be 0 from shard to itself
                    np.zeros((numOfMachines,numOfMachines)) <= X, # Can not have negative nodes moving
#                     S - V <= (X - X.T - cvx.diag(cvx.diag(X-X.T)))*ones_vector,
#                     (X - X.T - cvx.diag(cvx.diag(X-X.T)))*ones_vector <= np.maximum(zeros_vector, T - V)]
                    S - V <= (X.T - X - cvx.diag(cvx.diag(X.T-X)))*ones_vector,
                    (X.T - X - cvx.diag(cvx.diag(X.T-X)))*ones_vector <= np.maximum(zeros_vector, T - V)]

    prob = cvx.Problem(objective, constraints + messyConstraint)

    # ECOS doesn't crash with >2 shards like CVXOPT does...
    opt = prob.solve(solver='ECOS')
    
    if prob.status not in ['infeasible', 'unbounded']:
        variables = {}
        for i in range(numOfMachines):
            for j in range(numOfMachines):
                variables[(i,j)] = int(np.round(X.value[i,j]))
    else:
        # If infeasible/unbounded, return previous assignment
        variables = {}
        for i in range(numOfMachines):
            for j in range(numOfMachines):
                variables[(i,j)] = 0
    return variables
