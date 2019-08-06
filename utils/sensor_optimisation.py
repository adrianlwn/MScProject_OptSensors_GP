import numpy as np
import scipy

import sys
import os
from os.path import join
import pandas as pd
import time

import tqdm
import heapq
import pdb


def define_sets(V, n_S=None, seed=42):
    """ Function that defines the sets used in the optimsation problem. S is randomly selected in V.
        Inputs :
        --- V : idx of the set of selected points of the dataset
        --- n_S : number of sensor to randomly selected in V (if None, S = V )
        --- seed : seed used for the random selection of S
        Outputs :
        --- n_V, V : size and idx of points contained in V : set of all points
        --- n_S, S : size and idx of points contained in S : set of potential sensor points
        --- n_U, U : size and idx of points contained in U : set of non sensor points

    """

    n_V = len(V)

    if n_S == None:
        n_S = n_V
        S = V
    else:
        np.random.seed(seed)
        S = np.random.choice(V, n_S, replace=False, )

    n_U = n_V - n_S
    U = np.setdiff1d(V, S, assume_unique=True)

    return n_V, V, n_S, S, n_U, U


def sensor_loc_optimisation_naive(k, K, sets):
    """ This function implements the Algorithm 1: Approximation algorithm
    for maximizing mutual information.

    Input Arguments :
    --- k : number of Sensors to place in S ( k <= n_S )
    --- sets : the ensemble of sets

    Needed global variables :
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points

    """
    n_V, V, n_S, S, n_U, U = sets
    
    p2i = dict(zip(S.tolist(),range(0,len(S))))
    i2p = dict(zip(range(0,len(S)),S.tolist()))

    n_A = 0
    A = np.array([])
    
    delta_y = dict(zip(S.flatten().tolist(), np.zeros(n_S).tolist()))

    # Main Loop of the Algorithm : Iterating over the number of sensors to place

    for j in tqdm.tqdm_notebook(range(k)):
        # Get the Indexes of the Points
        
        S_A = np.setdiff1d(S, A).astype(int)
                
        ## Inner Loop : Iterating over the potential sensor places
        for i, y in enumerate(S_A):
            A_ = np.setdiff1d(V, np.append(A, [y]))
            
            # Mutual Information Gain
            delta_y[y] = H_cond(y, A, K, p2i, i2p) / H_cond(y, A_, K, p2i, i2p)

        # Greedily selection the best point to add to A
        y_opt = max(delta_y, key=delta_y.get)
        delta_y.pop(y_opt)

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)

    return A


def sensor_loc_optimisation_lazy(k, K, sets):
    """ This function implements the Algorithm 2: Approximation algorithm for
    maximizing mutual information efficiently using lazy evaluation.

    Input Arguments :
    --- k : number of Sensors to place
    --- sets : the ensemble of sets

    Needed global variables :
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points

    """
    n_V, V, n_S, S, n_U, U = sets
    
    p2i = dict(zip(S.tolist(),range(0,len(S))))
    i2p = dict(zip(range(0,len(S)),S.tolist()))


    # INIT :
    n_A = 0
    A = np.array([])

    delta_y = -1 * np.inf * np.ones((n_S, 1))
    counter_y = -1 * np.ones((n_S, 1))

    # Each Node of the Heap contains a tupple : (-delta_y, index of point, count_y)
    delta_heap = list(zip(delta_y, S, counter_y))
    heapq.heapify(delta_heap)

    # MAIN LOOP of the Algorithm : Iterating over the number of sensors to place
    for j in tqdm.tqdm(range(k),desc="Main Loop : Sensor Placement"):
        
        ## INNER LOOP : Iterating over the potential sensor places
        while True:
            delta, y_opt, count = heapq.heappop(delta_heap)
            if count == j:
                break
            else:
                A_ = np.setdiff1d(V, np.append(A, [y_opt]))
                
                # Mutual Information Gain
                delta_y_opt = H_cond(y_opt, A, K, p2i, i2p) / H_cond(y_opt, A_, K, p2i, i2p)
                heapq.heappush(delta_heap, (-1 * delta_y_opt, y_opt, j))

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
    return A


def sensor_loc_optimisation_naive_local(k, K, method, param, sets):
    """ NOT FININISHED : This function implements the Algorithm 3: Approximation algorithm for
    maximizing mutual information efficiently using local kernels.

    Input Arguments :
    --- k : number of Sensors to place
    --- epsilon : threshold for local kernel

    Needed global variables :
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points

    """

    if method == "threshold":
        epsilon = param
    elif method == "fixed":
        d = param
    else:
        print("Specifiy a method such as threshold or fixed ")
        return -1

    n_V, V, n_S, S, n_U, U = sets
    
    p2i = dict(zip(S.tolist(),range(0,len(S))))
    i2p = dict(zip(range(0,len(S)),S.tolist()))

    n_A = 0
    A = np.array([])

    # Initialisation Loop : 
    delta_y = dict(zip(S.flatten().tolist(), np.zeros(n_S).tolist()))
    for y in tqdm.tqdm(S,desc="Init Loop"):
        A_ = np.setdiff1d(V, np.append(A, [y]))
        

        # Mutual Information Gain
        if method == "threshold":
            delta_y[y] = H_cond(y, A, K, p2i, i2p) / H_cond_epsilon(y, A_, K, epsilon, p2i, i2p)

        elif method == "fixed":
            delta_y[y] = H_cond(y, A, K, p2i, i2p) / H_cond_fixed(y, A_, K, d, p2i, i2p)


    # Main Loop of the Algorithm : Iterating over the number of sensors to place
    S_A = np.setdiff1d(S, A).astype(int)
    for j in tqdm.tqdm(range(k),desc="Main Loop : Sensor Placement"):

        # Greedily selection the best point to add to A
        y_opt = max(delta_y, key=delta_y.get)
        delta_y.pop(y_opt)

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)

        S_A = np.setdiff1d(S, A).astype(int)

        # Set of most correlated points
        if method == "threshold":
            N_yopt_espilon = local_set(y_opt, S_A, K, epsilon, p2i ,i2p)

        elif method == "fixed":
            N_yopt_espilon = local_set_fixed(y_opt, S_A, K, d, p2i ,i2p)

        ## Inner Loop : Iterating over the potential sensor places
        for y in tqdm.tqdm(S_A,desc="Inner Loop : placing sensor " + str(j+1)):
            if y in N_yopt_espilon:
                A_ = np.setdiff1d(V, np.append(A, [y]))
                if method == "threshold":
                    delta_y[y] = H_cond_epsilon(y, A, K, epsilon, p2i, i2p) / H_cond_epsilon(y, A_, K, epsilon, p2i, i2p)

                elif method == "fixed":
                    delta_y[y] = H_cond_fixed(y, A, K, d, p2i, i2p) / H_cond_fixed(y, A_, K, d, p2i, i2p)
    return A


def delta_MI(y, A, K):
    """ Function that computes the Mutual Entropy Difference """
    A_ = np.setdiff1d(V, np.append(A, [y]))

    return H_cond(y, A, K) / H_cond(y, A_, K)


def H_cond(y, X, K, p2i, i2p):
    """ Function that returns the conditional Entropy of y knowing X """
    iX = list(map(p2i.__getitem__, X))
    iy = p2i[y]
    return K[iy, iy] - K[np.ix_([iy], iX)] @ np.linalg.inv(K[np.ix_(iX, iX)]) @ K[np.ix_(iX, [iy])]


def H_cond_epsilon(y, X, K, epsilon, p2i, i2p):
    """ Function that returns the conditional Entropy of y knowing X """
    iy = p2i[y]
    X_trunc = local_set(y, X, K, epsilon,p2i,i2p)
    iX_trunc = list(map(p2i.__getitem__, X_trunc))
    return K[iy, iy] - K[np.ix_([iy], iX_trunc)] @ np.linalg.inv(K[np.ix_(iX_trunc, iX_trunc)]) @ K[
        np.ix_(iX_trunc, [iy])]

def H_cond_fixed(y, X, K, d, p2i, i2p):
    """ Function that returns the conditional Entropy of y knowing X """
    iy = p2i[y]
    X_trunc = local_set_fixed(y, X, K, d,p2i,i2p)
    iX_trunc = list(map(p2i.__getitem__, X_trunc))
    return K[iy, iy] - K[np.ix_([iy], iX_trunc)] @ np.linalg.inv(K[np.ix_(iX_trunc, iX_trunc)]) @ K[
        np.ix_(iX_trunc, [iy])]


def local_set(y, X, K, epsilon, p2i, i2p):
    """ Function that returns a set of points X_trunc for which K[y,X_trunc] > epsilon
        X being the input set
        Implementing the idea of local Kernels.

    """
    iX = list(map(p2i.__getitem__, X))
    iy = p2i[y]
    i_trunc = (np.abs(K[np.ix_([iy], iX)]) > epsilon).flatten()

    iX_trunc = np.array(iX)[i_trunc]
    X_trunc = list(map(i2p.__getitem__, iX_trunc))
    print('Length of local covariance set : d = ', len(X_trunc))

    return X_trunc


def local_set_fixed(y, X, K, d, p2i, i2p):
    """ Function that returns a set of d points X_trunc for which only the d most correlated points are taken.
        Implementing the idea of local Kernels.

    """
    iX = list(map(p2i.__getitem__, X))
    iy = p2i[y]
    
    i_trunc = np.argsort(K[np.ix_([iy], iX)].flatten())[::-1][:d].tolist()
    iX_trunc = np.array(iX)[i_trunc]
    X_trunc = list(map(i2p.__getitem__, iX_trunc))
    print('Length of local covariance set : d = ', len(X_trunc), ", largest = ", K[np.ix_([iy], [np.array(iX)[i_trunc[0]]])]," smallest =", K[np.ix_([iy], [np.array(iX)[i_trunc[-1]]])])        
    #pdb.set_trace()


    return X_trunc
