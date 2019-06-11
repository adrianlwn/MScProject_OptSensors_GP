import numpy as np
import sys
from os.path import join
import pandas as pd
import time
from tqdm import tqdm

from utils.config import *

sys.path.append('../fluidity-master/python/')
import vtktools


def load_vtu(ts_0, ts_end, crop=None, ):
    """ Function that loads the VTK files in memory from timestamps : ts_0 to ts_end. 
        It crops the data if required.
        
        Inputs : 
        --- ts_0 : initial time stamp, must be in [0 -> 988] 
        --- ts_end : final time stamp, must be in [0 -> 988] and ts_end > ts_0
        --- crop : No cropping of the space : None
                   Cropping of the space : ((min_x , max_x),(min_y, max_y),(min_z, max_z))
        Returns :
        --- data_dict : dictionary indexed by time stamp of all the loaded VTK files (all fields)
        --- location_df : a pandas DataFrame of the location of space
        --- time_vec : a numpy array containg the real time stamps of the simulation 
    
    """

    print("==> Import vtu files from {} to {}".format(ts_0, ts_end))
    data_dict = {}
    for ts in tqdm(range(ts_0, ts_end + 1)):
        try:
            full_path = join(data_path, VTK_folder)
            file_path = join(full_path, 'LSBU_' + str(ts) + '.vtu')
            data_dict[ts] = vtktools.vtu(file_path)
        except:
            print('Can\'t open the file \'' + file_path + '\'')

    print('Number of Locations : ', data_dict[ts_0].GetLocations().shape[0])

    if crop is not None:
        print("==> Cropping vtu files to {}".format(str(crop)))

        crop_data(data_dict,
                  min_x=crop[0][0], max_x=crop[0][1],
                  min_y=crop[1][0], max_y=crop[1][1],
                  min_z=crop[2][0], max_z=crop[2][1])

        print('Number of Locations after cropping : ', data_dict[ts_0].GetLocations().shape[0])

    ref_vtu = data_dict[ts_0]
    return data_dict, ref_vtu, extract_location_df(data_dict), extract_time_vec(data_dict)


def save_vtu(ref_vtu, field_name, field_data):
    """ Function that saves all the fields specified in the list
        Input : 
        --- data_dict : dictionary indexed by time stamp of all the loaded VTK files (all fields)
        --- field_name : string or list of strings of the field names
        --- field_data : np.array of the fields to save - shape : (n_locations, n_fields)
        
    """

    if type(field_name) is str:
        field_name = [field_name]

    # Load the first element of data_dict
    #ref_vtu = data_dict[list(data_dict.keys())[0]]

    # Copy this vtk file to the saving location

    time_str = time.strftime("%Y:%m:%d-%H:%M:%S")
    file_name = '_'.join(['LSBU_res', time_str, '_'.join(field_name)])

    file_path = join(join(data_path, temp_folder), file_name)
    ref_vtu.Write(file_path)

    # Load the copy of the file
    temp_vtk = vtktools.vtu(file_path)
    print('Saving to : {}'.format(temp_vtk.filename))

    # Add new fields 
    for i, name in field_name:
        vtktools.vtu.AddField(name, field_data[:, i])

    # Write those fields int the file : 
    temp_vtk.Write(file_path)


def crop_data(data_dict, min_x=-50, max_x=50, min_y=-50, max_y=50, min_z=0, max_z=50):
    """ Crops the space of the positions for each timestamp

    """

    for t, ts in tqdm(enumerate(data_dict)):
        data_dict[ts].Crop(min_x, max_x,
                           min_y, max_y,
                           min_z, max_z)


def extract_data_mat(data_dict, field_name):
    # Extracts the time series for the specific field at each position

    print("==> Extracting the field \"{}\" from the vtu files.".format(field_name))
    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]

    data_mat = np.zeros((N, T))

    for t, ts in enumerate(data_dict):
        data_mat[:, t] = data_dict[ts].GetField(field_name).T
    return data_mat


def extract_time_vec(data_dict):
    # Extracts the time vector of the simulation
    # Returns a numpy array

    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]

    time_vec = np.zeros((T, 1))

    for t, ts in enumerate(data_dict):
        time_vec[t] = data_dict[ts].GetField('Time')[0]
    return time_vec


def extract_data_df(data_dict, field_name):
    # Extracts the time series for the specific field at each position
    # Returns a pandas DataFrame

    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]

    data_mat = np.zeros((N, T))

    for t, ts in enumerate(data_dict):
        data_mat[:, t] = data_dict[ts].GetField(field_name).T
    return pd.DataFrame(data_mat)


def extract_location_df(data_dict):
    """ Function that returns the location of all the points in a dataframe with columns : ['X','Y','Z']
        Inputs : 
        --- data_dict : dictionary indexed by time stamp of all the loaded VTK files (all fields)
        Returns : 
        --- location_df : df of the points location in the mesh
        
    """
    i_0 = list(data_dict.keys())[0]
    return pd.DataFrame(data_dict[i_0].GetLocations(), columns=['X', 'Y', 'Z'])


def find_nearest_point(location_df, point=[0.0, 0.0, 0.0]):
    """ Function that returns the index and the location of the closest of the mesh.
        Closest point found by L2 distance.
        Inputs : 
        --- location_df : df of the points location in the mesh
        --- point : point that we are looking for in the mesh
        Returns :
        --- i_point : index of the point
        --- n_point : coordinates of the point
        
    """
    dist_df = location_df.apply(lambda x: np.linalg.norm([x['X'], x['Y'], x['Z']] - point), axis=1);
    i_point = dist_df.idxmin()
    n_point = location_df.loc[i_point, :].values
    return i_point, n_point


def idx_slice(location_df, direction='Z', s_min=0.0, s_max=1.0):
    """ Function that computes the slices of the mesh indexes in the direction and of the width specified.
        Input : 
        --- location_df : df of the points location in the mesh
        --- direction : direction in which we cut the space
        --- s_min : beginning of the slice 
        --- s_max : end of the splice : s_min < s_max
        
    """

    index_slice = location_df.loc[(s_min <= location_df[direction]) & (location_df[direction] < s_max), :].index

    return np.array(index_slice)


def idx_all_slices(location_df, direction='Z', w_slice=1.0):
    """ Function that computes all the slices of the mesh indexes in the direction and of the width specified.
        Input : 
        --- location_df : df of the points location in the mesh
        --- direction : direction in which we cut the space
        --- w_slice : width of the splice of space
        
    """

    d_min = location_df[[direction]].min().values
    d_max = location_df[[direction]].max().values
    r_min = d_min - (d_min % w_slice)
    r_max = d_max + w_slice

    bins = np.arange(r_min, r_max, w_slice)

    index_slices_dict = location_df[[direction]].apply(lambda x: bins[np.digitize(x, bins, right=False)]).groupby(
        direction).apply(lambda x: x.index.tolist()).to_dict()

    return index_slices_dict
