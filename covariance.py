import numpy as np
import sys
import os
from os.path import join
import pandas as pd
import time


import tqdm
import heapq

temp_path = "../data/temp_data/"

def cov_matrix(X):
    """ Computes the covariance matrix of the input mat
    
    """
    
    K = np.cov(X)
    return K

def save_matrix(X,name, inplace= False):
    """ Function that saves in .npy format the X matrix with the specified name
    
    """
    
    # Define the Path
    full_path = join(temp_path,'_'.join([name,str(0)]) + '.npy')
    
    # Check if file needs to be replaced
    if not(inplace) : 
        i = 0
        while os.path.isfile(full_path):
            i += 1
            full_path = join(temp_path,'_'.join([name,str(i)]) + '.npy')
            
    # Save the File
    np.save(full_path,X)
    print('Saved to \'', full_path, '\'')
            
def load_matrix(name, n = None):
    """ Function that loads the matrix file specified and returns it. 
        Inputs : 
        --- name : name of the file without extension
        --- n : itteration of the file (if None, the last file will be loaded)
        
    """
    X = []
    if n == None :
        i = 0
        full_path = join(temp_path,'_'.join([name,str(i)]) + '.npy')
        while os.path.isfile(full_path):
            i += 1
            full_path = join(temp_path,'_'.join([name,str(i)]) + '.npy')
        full_path = join(temp_path,'_'.join([name,str(i-1)]) + '.npy')
        
    else :       
        full_path = join(temp_path,'_'.join([name,str(n)]) + '.npy')
        
    
    try:
        X = np.load(full_path)
        print('Loaded from \'', full_path, '\'')

    except:
        print('Can\'t open the file \'' + full_path + '\'')
    
    return X


def cov_matrix_p(X, step_size = 1e5, n_processes = 100):
    ''' Computes the covariance matrix in parallel '''
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    v = pd.DataFrame({'index': range(n_samples)})
    
    # index array for parallelization
    pos_array = np.array(np.linspace(0, n_features*(n_features-1)//2, n_processes), dtype=int)
    
    # connector function to compute pairwise pearson correlations
    def cov(index_s, index_t):
        features_s = X[index_s]
        features_t = X[index_t]
        cov = np.einsum('ij,ij->i', features_s, features_t) / n_samples
        return cov

    return K


def mean_matrix(data_mat):
    mean_mat = np.mean(data_mat,axis=1)
    return mean_mat
