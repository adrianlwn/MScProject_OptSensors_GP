import numpy as np
import sys
import os
from os.path import join
import pandas as pd



sys.path.append('../fluidity-master/python/')
import vtktools


import tqdm
import heapq

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data_path = "../data/small3DLSBU/"

def load_data(ts_0,ts_end):
    data_dict = {}
    for ts in range(ts_0,ts_end+1):
        try:
            data_dict[ts] = vtktools.vtu(join(data_path,'LSBU_'+ str(ts)+'.vtu'))
        except:
            print('Can\'t open the file \'' + str(join(data_path,'LSBU_'+ str(ts)+'.vtu'))+ '\'')
    return data_dict

def crop_data(data_dict,min_x= -50, max_x= 50, min_y= -50, max_y= 50, min_z=0, max_z=50):
    ''' Crops the space of the positions for each timestamp '''
    for t, ts in enumerate(data_dict) :        
        cropped = data_dict[ts].Crop(min_x , max_x,
                                     min_y, max_y,
                                     min_z, max_z)
    
    
def to_matrix(data_dict,field_name):
    ''' Extracts the time series for the specific field at each position'''
    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]
            
    data_mat = np.zeros((N,T))
            
    for t, ts in enumerate(data_dict) :
            data_mat[:,t] = data_dict[ts].GetField(field_name).T
    return data_mat

def cov_matrix(X):
    ''' Computes the covariance matrix using np.cov function - Not parallel'''
    K = np.cov(X)
    return K


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

            
            
        