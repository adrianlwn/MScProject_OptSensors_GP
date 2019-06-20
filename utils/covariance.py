import numpy as np
import os
from os.path import join, isfile
import pandas as pd

from utils.config import *
from utils.data_handling import create_temp_folder




def cov_matrix(X, parameters, recompute=False):
    """ Main function that computes/loads and saves the covariance matrix for X
        according to the specified method in parameters.
s
    :param X: data of computed covariance
    :param parameters: dict of param
    :param recompute: if we force to recompute the
    :return:
    """
    method = parameters["cov_method"]

    # Check if covariance is already computed
    folder_path = create_temp_folder(parameters)
    cov_file_name = "_".join(["cov", method]).npy
    cov_full_path = join(folder_path, cov_file_name)

    if (not recompute) & isfile(cov_full_path):
        K = np.load(cov_full_path)
        print('Loaded from \'', cov_full_path, '\'')
    else:
        recompute = True

    # Computing the Covariance matrix

    if recompute & method == "sample":
        K = sample_cov_matrix(X)
        # Saving it in file
        np.save(cov_full_path, K)
        print('Saved to \'', cov_full_path, '\'')

    elif recompute & method == "exponential_kernel":

        pass


    elif recompute:
        print("Incorrect method")
    # Saving the Covariance Matrix in File

    return K




def sample_cov_matrix(X):
    """ Computes the sample covariance matrix using numpy function
    """
    return np.cov(X)

def exponential_kernel(r, l):
    """ This function defines the exponential kernal (p.83 of GP Book)
        Inputs : 
        --- r : distance between 2 points
        --- l : lengthscale that parametrize the strength of the covariance
    """
    return np.exp(- r ** 2 / (2 * l ** 2))


def matern32_kernel(r, l):
    """ This function defines the matern 3/2 kernal (p.83 of GP Book)
        Inputs : 
        --- r : distance between 2 points
        --- l : lengthscale that parametrize the strength of the covariance
    """

    return (1 + np.sqrt(3) * r / l) * np.exp(- np.sqrt(3) * r / l)


def matern52_kernel(r, l):
    """ This function defines the matern 5/2 kernal (p.85 of GP Book)
        Inputs : 
        --- r : distance between 2 points
        --- l : lengthscale that parametrize the strength of the covariance
    """

    return (1 + np.sqrt(5) * r / l + 5 * r ** 2 / (3 * l ** 2)) * np.exp(- np.sqrt(5) * r / l)
