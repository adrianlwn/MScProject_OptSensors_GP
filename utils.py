import numpy as np
import sys
from os.path import join


sys.path.append('../fluidity-master/python/')
import vtktools


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
    for t, ts in enumerate(data_dict) :        
        cropped = data_dict[ts].Crop(min_x , max_x,
                                     min_y, max_y,
                                     min_z, max_z)
    
    
def to_matrix(data_dict,field_name):
    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]
            
    data_mat = np.zeros((N,T))
            
    for t, ts in enumerate(data_dict) :
            data_mat[:,t] = data_dict[ts].GetField(field_name).T
    return data_mat
            
            
        