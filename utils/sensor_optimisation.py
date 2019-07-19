import numpy as np
import scipy

import sys
import os
from os.path import join
import pandas as pd
import time

import tqdm
import heapq


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

    n_A = 0
    A = np.array([])

    # Main Loop of the Algorithm : Iterating over the number of sensors to place

    for j in tqdm.tqdm_notebook(range(k)):

        S_A = np.setdiff1d(S, A).astype(int)
        delta_y = np.zeros((n_S - j))
        ## Inner Loop : Iterating over the potential sensor places
        for i, y in enumerate(S_A):
            A_ = np.setdiff1d(V, np.append(A, [y]))
            # Mutual Information Gain

            delta_y[i] = H_cond(y, A, K) / H_cond(y, A_, K)

        # Greedily selection the best point to add to A
        y_opt = S_A[np.argmax(delta_y)]

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

    # INIT :
    n_A = 0
    A = np.array([])

    delta_y = -1 * np.inf * np.ones((n_S, 1))
    counter_y = -1 * np.ones((n_S, 1))

    # Each Node of the Heap contains a tupple : (-delta_y, index of point, count_y)
    delta_heap = list(zip(delta_y, S, counter_y))
    heapq.heapify(delta_heap)

    # MAIN LOOP of the Algorithm : Iterating over the number of sensors to place
    for j in tqdm.tqdm_notebook(range(k)):

        ## INNER LOOP : Iterating over the potential sensor places
        while True:
            delta, y_opt, count = heapq.heappop(delta_heap)
            if count == j:
                break
            else:
                A_ = np.setdiff1d(V, np.append(A, [y_opt]))
                # Mutual Information Gain
                delta_y_opt = H_cond(y_opt, A, K) / H_cond(y_opt, A_, K)
                heapq.heappush(delta_heap, (-1 * delta_y_opt, y_opt, j))

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
    return A


def sensor_loc_optimisation_naive_truncated(k, K, epsilon):
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

    n_V, V, n_S, S, n_U, U = sets

    n_A = 0
    A = np.array([])
    
    # Initialisation Loop : 
    delta_y = np.zeros(n_S)
    for i, y in enumerate(S):
            A_ = np.setdiff1d(V, np.append(A, [y]))
            # Mutual Information Gain

            delta_y[i] = H_cond(y, A, K) / H_cond_epsilon(y, A_, K, epsilon)
            
    

    # Main Loop of the Algorithm : Iterating over the number of sensors to place
    S_A = np.setdiff1d(S, A).astype(int)
    for j in tqdm.tqdm_notebook(range(k)):
        
         # Greedily selection the best point to add to A
        i_y_opt = np.argmax(delta_y)
        delta_y = np.remove(delta_y,i_y_opt)
        y_opt = S_A[i_y_opt]

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)

        S_A = np.setdiff1d(S, A).astype(int)
        
        ## Inner Loop : Iterating over the potential sensor places
        N_yopt_espilon = local_set(y_opt, S_A, K, epsilon)
        for i, y in enumerate(S_A):
            if y in N_yopt_espilon : 
                A_ = np.setdiff1d(V, np.append(A, [y]))
                # Mutual Information Gain

                delta_y[i] = H_cond_epsilon(y, A, K, epsilon) / H_cond_epsilon(y, A_, K, epsilon)
    return A


def sensor_loc_optimisation_naive_truncated_adaptative(k, K, d):
    """ NOT FININISHED : This function implements the Algorithm 3: Approximation algorithm for
    maximizing mutual information efficiently using local kernels.

    Input Arguments :
    --- k : number of Sensors to place
    --- d : number of correlated points to keep. the parameter epsilon adapts itself to d 

    Needed global variables :
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points

    """

    n_V, V, n_S, S, n_U, U = sets

    n_A = 0
    A = np.array([])
    
    # Initialisation Loop : 
    delta_y = np.zeros(n_S)
    for i, y in enumerate(S):
            A_ = np.setdiff1d(V, np.append(A, [y]))
            # Mutual Information Gain

            delta_y[i] = H_cond(y, A, K) / H_cond_epsilon(y, A_, K, epsilon)
            
    

    # Main Loop of the Algorithm : Iterating over the number of sensors to place
    S_A = np.setdiff1d(S, A).astype(int)
    for j in tqdm.tqdm_notebook(range(k)):
        
         # Greedily selection the best point to add to A
        i_y_opt = np.argmax(delta_y)
        delta_y = np.remove(delta_y,i_y_opt)
        y_opt = S_A[i_y_opt]

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)

        S_A = np.setdiff1d(S, A).astype(int)
        
        ## Inner Loop : Iterating over the potential sensor places
        N_yopt_espilon = local_set(y_opt, S_A, K, epsilon)
        for i, y in enumerate(S_A):
            if y in N_yopt_espilon : 
                A_ = np.setdiff1d(V, np.append(A, [y]))
                # Mutual Information Gain

                delta_y[i] = H_cond_epsilon(y, A, K, epsilon) / H_cond_epsilon(y, A_, K, epsilon)
    return A



def delta_MI(y, A, K):
    """ Function that computes the Mutual Entropy Difference """
    A_ = np.setdiff1d(V, np.append(A, [y]))

    return H_cond(y, A, K) / H_cond(y, A_, K)


def H_cond(y, X, K):
    """ Function that returns the conditional Entropy of y knowing X """
    return K[y, y] - K[np.ix_([y], X)] @ np.linalg.inv(K[np.ix_(X, X)]) @ K[np.ix_(X, [y])]

def H_cond_epsilon(y, X, K, epsilon):
    """ Function that returns the conditional Entropy of y knowing X """
    X_truncated = local_set(y, X, K, epsilon)
    return K[y, y] - K[np.ix_([y], X_truncated)] @ np.linalg.inv(K[np.ix_(X_truncated, X_truncated)]) @ K[np.ix_(X_truncated, [y])]


def local_set(y, X, K, epsilon):
    """ Function that returns a set of points X_trunc for which K[y,X_trunc] > epsilon
        X being the input set
        Implementing the idea of local Kernels.

    """
    i_trunc = (np.abs(K[np.ix_([y], X)]) > epsilon).flatten()
    X_trunc = X[i_trunc]
    print('Length of local covariance set : d = ', len(X_trunc))

    return X_trunc


def local_set_fixed(y, X, K, d):
    """ Function that returns a set of points X_trunc for which K[y,X_trunc] > epsilon
        X being the input set with |K[y,X_trunc]| <= d
        Implementing the idea of local Kernels.

    """
    truncated = True
    epsilon = 0.1
    decay_rate = 0.1
    while True
    i_trunc = (np.abs(K[np.ix_([y], X)]) > epsilon).flatten()
    X_trunc = X[i_trunc]
    print('Length of local covariance set : d = ', len(X_trunc))
    
    epsilon *= decay_rate
    
    

    return X_trunc
