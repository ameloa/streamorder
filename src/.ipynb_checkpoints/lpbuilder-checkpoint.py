import sys
import cvxpy as cvx
import numpy as np
from math import floor, ceil

# rL: request list. sorted list of lists. each list corresponds to a move request. (i, j, improvement, # of identical requests). Sorted in order of i, then j, then descending improvement.
# moverMap: dictionary. key = (i,j). value = list of (node, improvement) tuples.
# populationCount: dictionary. key = shard. value = population size.
'''
    Implementation of the linear program of Balanced Label Propagation. 
    For implementation details, see the original paper, https://stanford.edu/~jugander/papers/wsdm13-blp.pdf
'''
def solve_lp(numNodes, numShards, rL, populationCount, epsilon):
    ones_vector = np.ones(numShards)
    zeros_vector = np.zeros(numShards)
    S, T, V = np.ones(numShards), np.ones(numShards), np.ones(numShards)
    for i in range(numShards):
        S[i] = floor((1.0 - epsilon) * (float(numNodes)/float(numShards)))
        T[i] = ceil((1.0 + epsilon) * (float(numNodes)/float(numShards)))
        V[i] = populationCount[i]

    # Construct the problem.
    Z = cvx.Variable((numShards, numShards)) # Objective
    X = cvx.Variable((numShards, numShards)) # Number of nodes to move

    messyConstraint = []
    ix = 0 # index of move request
    for i in range(numShards):
        jrange = list(range(numShards))
        jrange.remove(i)
        for j in jrange:
            a = {}
            b = {}
            Sum = {}
            b[0] = 0
            Sum[-1] = 0
            if (ix > len(rL) - 1):
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
                    np.zeros((numShards,numShards)) <= X, # Can not have negative nodes moving
                    S - V <= (X.T - X - cvx.diag(cvx.diag(X.T-X)))@ones_vector,
                    (X.T - X - cvx.diag(cvx.diag(X.T-X)))@ones_vector <= np.maximum(zeros_vector, T - V)]

    prob = cvx.Problem(objective, constraints + messyConstraint)

    # ECOS doesn't crash with >2 shards like CVXOPT does...
    opt = prob.solve(solver='ECOS')
    
    if prob.status not in ['infeasible', 'unbounded']:
        variables = {}
        for i in range(numShards):
            for j in range(numShards):
                variables[(i,j)] = int(np.round(X.value[i,j]))
    else:
        # If infeasible/unbounded, return previous assignment
        variables = {}
        for i in range(numShards):
            for j in range(numShards):
                variables[(i,j)] = 0
    return variables
