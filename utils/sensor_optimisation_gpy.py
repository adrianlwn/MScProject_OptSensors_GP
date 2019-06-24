import numpy as np
import sys
import os
from os.path import join
import pandas as pd
import time

import tqdm
import heapq

import GPy


def define_sets(n_V, n_S=None, seed=42):
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

    V = np.array(range(n_V))

    if n_S == None:
        n_S = n_V
        S = V
    else:
        np.random.seed(seed)
        S = np.random.choice(V, n_S, replace=False, )

    n_U = n_V - n_S
    U = np.setdiff1d(V, S, assume_unique=True)

    return n_V, V, n_S, S, n_U, U


def approx_max_info(k, K, sets):
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


def H_cond(y, X, K):
    """ Function that returns the conditional Entropy of y knowing X """
    return K[y, y] - K[np.ix_([y], X)] @ np.linalg.inv(K[np.ix_(X, X)]) @ K[np.ix_(X, [y])]


def H_cond_gpflow(y, X):
    """ Function that returns the conditional Entropy of y knowing X """


    return K[y, y] - K[np.ix_([y], X)] @ np.linalg.inv(K[np.ix_(X, X)]) @ K[np.ix_(X, [y])]