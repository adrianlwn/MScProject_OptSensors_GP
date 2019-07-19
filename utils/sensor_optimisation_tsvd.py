import numpy as np
import scipy

import sys
import os
from os.path import join
import pandas as pd
import time

import tqdm
import heapq
from sklearn.decomposition import TruncatedSVD


def define_sets(V, n_S=None, seed=42):
    """ Function that defines the sets used in the optimsation problem. S is randomly selected in V.
        Inputs :
        --- n_V : number of points in the datasets
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


def sensor_loc_optimisation_naive(k, Z, sets, tau):
    """ This function implements the Algorithm 1: Approximation algorithm
    for maximizing mutual information.

    Input Arguments :
    --- k : number of Sensors to place in S ( k <= n_S )
    --- sets : the ensemble of sets

    Needed global variables :
    --- Z
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points
    """
    
    n_V, V, n_S, S, n_U, U = sets

    n_A = 0
    A = np.array([])

    # Main Loop of the Algorithm : Iterating over the number of sensors to place

    for j in tqdm.tqdm(range(k), desc="Main Loop"):

        S_A = np.setdiff1d(S, A).astype(int)
        delta_y = np.zeros((n_S - j))
        ## Inner Loop : Iterating over the potential sensor places
        for i, y in  tqdm.tqdm(enumerate(S_A), desc="Inner Loop"):
            A_ = np.setdiff1d(V, np.append(A, [y]))
            # Mutual Information Gain

            delta_y[i] = H_cond(y, A, Z, tau) / H_cond(y, A_, Z, tau)

        # Greedily selection the best point to add to A
        y_opt = S_A[np.argmax(delta_y)]

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
        print(A)

    return A


def sensor_loc_optimisation_lazy(k, Z, sets, tau):
    """ This function implements the Algorithm 2: Approximation algorithm for
    maximizing mutual information efficiently using lazy evaluation.

    Input Arguments :
    --- k : number of Sensors to place
    --- sets : the ensemble of sets

    Needed global variables :
    --- Z : 
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
    for j in tqdm.tqdm(range(k)):
        p_bar = tqdm.tqdm(range(n_S), desc="Inner Loop")
        ## INNER LOOP : Iterating over the potential sensor places
        while True:
            p_bar.update(1)
            delta, y_opt, count = heapq.heappop(delta_heap)
            if count == j:
                break
            else:
                A_ = np.setdiff1d(V, np.append(A, [y_opt]))
                # Mutual Information Gain
                delta_y_opt = H_cond(y_opt, A, Z, tau) / H_cond(y_opt, A_, Z, tau)
                heapq.heappush(delta_heap, (-1 * delta_y_opt, y_opt, j))

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
        print(A)

    return A


def approx_local_max_info(k, K, epsilon):
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

    # INIT :
    n_A = 0
    A = np.array([])

    delta_y = -1 * np.inf * np.ones((n_S, 1))

    for i, y in enumerate(S):
        delta_y[i] = -1 * delta_MI(y, A)

    counter_y = -1 * np.ones((n_S, 1))

    # Each Node of the Heap contains a tupple : (-delta_y, index of point, count_y)
    delta_heap = list(zip(delta_y, S, counter_y))
    heapq.heapify(delta_heap)

    # MAIN LOOP of the Algorithm : Iterating over the number of sensors to place
    for j in tqdm.tqdm(range(k)):

        delta, y_opt, count = heapq.heappop(delta_heap)

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
        A_ = np.setdiff1d(V, np.append(A, [y]))

        loc_A = local_set(y, A, epsilon)
        loc_A_ = local_set(y, A_, epsilon)

        ## INNER LOOP : Iterating over the potential sensor places
        for i in A:
            # Mutual Information Gain
            delta_y = H_cond(y, loc_A, K) / H_cond(y, loc_A_, K)

            heapq.heappush(delta_heap, (-1 * delta_y, y, j))

    return A




def H_cond(y, X, Z, tau):
    """ Function that returns the conditional Entropy of y knowing X """
    
    Z_y = Z[y,:].reshape(1,-1)
    if X.shape[0] == 0 :
        Z_y_A = Z_y @ Z_y.T
    else :
        if len(X) <= tau:
            Z_Atau = Z[X,:].reshape(1,-1)
        else :
            tsvd = TruncatedSVD(n_components=tau)
            Z_Atau = tsvd.fit_transform(Z[X.astype(int),:].T).T
        
        Z_y_A  = Z_y @ Z_y.T -  Z_y @ Z_Atau.T @ np.linalg.inv(Z_Atau @ Z_Atau.T) @ Z_Atau @ Z_y.T


    return Z_y_A
    #return K[y, y] - K[np.ix_([y], X)] @ np.linalg.inv(K[np.ix_(X, X)]) @ K[np.ix_(X, [y])]
    # return K[y,y] - K[np.ix_([y],X)] @ np.linalg.solve(K[np.ix_(X,X)], K[np.ix_(X,[y])])


def local_set(y, X, K, epsilon):
    """ Function that returns a set of points X_trunc for which K[y,X_trunc] > epsilon
        X being the input set
        Implementing the idea of local Kernels.

    """
    i_trunc = (np.abs(K[np.ix_([y], X)]) > epsilon).flatten()
    X_trunc = X[i_trunc]

    return X_trunc
