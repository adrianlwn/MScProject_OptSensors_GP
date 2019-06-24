import numpy as np
import sys
import os
from os.path import join
import pandas as pd
import time

import tqdm
import heapq

import GPy


def define_sets(n_V, n_S=None):
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
        S = np.random.choice(V, n_S, replace=False, )

    n_U = n_V - n_S
    U = np.setdiff1d(V, S, assume_unique=True)

    return n_V, V, n_S, S, n_U, U


def sensor_loc_optimisation_naive(k, sets, X, Y, kernel):
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

    for j in tqdm.tqdm(range(k)):

        S_A = np.setdiff1d(S, A).astype(int)
        delta_y = np.zeros((n_S - j))
        ## Inner Loop : Iterating over the potential sensor places
        for i, y in enumerate(S_A):
            A_ = np.setdiff1d(V, np.append(A, [y]))
            # Mutual Information Gain

            delta_y[i] = delta_mi(y, A, A_, X, Y, kernel)

        # Greedily selection the best point to add to A
        y_opt = S_A[np.argmax(delta_y)]

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)

    return A


def sensor_loc_optimisation_lazy(k, sets, X, Y, kernel):
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

    # INIT :
    n_A = 0
    A = np.array([])

    delta_y = -1 * np.inf * np.ones((n_S, 1))
    counter_y = -1 * np.ones((n_S, 1))

    # Each Node of the Heap contains a tuple : (-delta_y, index of point, count_y)
    delta_heap = list(zip(delta_y, S, counter_y))
    heapq.heapify(delta_heap)

    # MAIN LOOP of the Algorithm : Iterating over the number of sensors to place
    for j in tqdm.tqdm(range(k)):

        ## INNER LOOP : Iterating over the potential sensor places
        while True:
            delta, y_opt, count = heapq.heappop(delta_heap)
            if count == j:
                break
            else:
                A_ = np.setdiff1d(V, np.append(A, [y_opt]))
                # Mutual Information Gain
                delta_y_opt = delta_mi(y_opt, A, A_, X, Y, kernel)
                heapq.heappush(delta_heap, (-1 * delta_y_opt, y_opt, j))

        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
    return A


def delta_mi(i_predict, A, A_, X, Y, kernel):
    """ Computes the mutual information difference before and after adding y to A

    :param i_predict:
    :param A:
    :param A_:
    :param m:
    :param kernel:
    :return:
    """
    x_predict = X[i_predict, :].reshape(1, -1)

    if A.shape[0] == 0:
        covA = kernel.K(x_predict)

    else:
        mA = GPy.models.GPRegression(X=X[A, :], Y=Y[A, :], kernel=kernel)
        _, covA = mA.predict_noiseless(x_predict)

    mA_ = GPy.models.GPRegression(X=X[A_, :], Y=Y[A_, :], kernel=kernel)
    _, covA_ = mA_.predict_noiseless(x_predict)
    return covA / covA_
