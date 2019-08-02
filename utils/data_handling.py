import numpy as np
import sys
from os import mkdir, listdir
from os.path import join, isdir, isfile
import pandas as pd
import time
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPoint


from utils.config import *

sys.path.append('../fluidity-master/python/')
import vtktools


# ============================================
# =============VTU files Handling=============
# ============================================

def initial_load_data(parameters, recompute=False):
    """ Function that loads the VTK files in memory from timestamps : ts_0 to ts_end. 
        It crops the data if required. It extracts the field required in parameters.
        
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
    ts_0 = parameters["i_start"]
    ts_end = parameters["i_end"]
    crop = parameters["crop"]

    # Path of the folder unique to the parameters
    data_folder = create_temp_folder(parameters)

    if not (recompute or not listdir(data_folder)):
        # If we don't have the files, we recompute them from vtu
        print("### Loading files from original VTU")
        ref_vtu = load_vtu(ts_0, ts_0, crop)
        loc_df = load_df('loc', parameters)
        time_df = load_df('time', parameters)
        data_df = load_df('data', parameters)

    else:
        print("### Loading preprocessed files")
        data_dict = load_vtu(ts_0, ts_end, crop)
        ref_vtu = data_dict[ts_0]

        loc_df = extract_location_df(data_dict)
        save_df('loc', loc_df, parameters)

        time_df = extract_time_df(data_dict)
        save_df('time', time_df, parameters)

        data_df = extract_data_df(data_dict, parameters)
        save_df('data', data_df, parameters)

    return ref_vtu, data_df, loc_df, time_df


def load_vtu(ts_0, ts_end, crop):
    """ Function that loads the vtu data in a dictionary

    :param ts_0:
    :param ts_end:
    :param crop:
    :return:
    """
    # If we already have the files, we import the content from the vtu files
    print("==> Import vtu files from {} to {}".format(ts_0, ts_end))
    if crop is not None:
        print("==> Cropping vtu files to {}".format(str(crop)))
    data_dict = {}
    for ts in tqdm(range(ts_0, ts_end + 1)):
        try:
            full_path = join(data_path, VTK_folder)
            file_path = join(full_path, 'LSBU_' + str(ts) + '.vtu')
            data_dict[ts] = vtktools.vtu(file_path)
        except:
            print('Can\'t open the file \'' + file_path + '\'')

        if crop is not None:
            data_dict[ts].Crop(crop[0][0], crop[0][1],
                               crop[1][0], crop[1][1],
                               crop[2][0], crop[2][1])

    print('Number of Locations after cropping : ', data_dict[ts_0].GetLocations().shape[0])

    return data_dict


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
    # ref_vtu = data_dict[list(data_dict.keys())[0]]

    # Copy this vtk file to the saving location

    time_str = time.strftime("%Y:%m:%d-%H:%M:%S")
    file_name = '_'.join(['LSBU_res', time_str, '_'.join(field_name)]) + ".vtu"

    file_path = join(join(data_path, temp_folder), file_name)
    ref_vtu.Write(file_path)

    # Load the copy of the file
    temp_vtk = vtktools.vtu(file_path)

    # Add new fields
    for i, name in enumerate(field_name):
        temp_vtk.AddField(name, field_data[:, i])

    # Write those fields int the file : 
    temp_vtk.Write(file_path)
    print('==> Saved to : {}'.format(temp_vtk.filename))


def crop_data(data_dict, min_x=-50, max_x=50, min_y=-50, max_y=50, min_z=0, max_z=50):
    """ Crops the space of the positions for each timestamp

    """

    for t, ts in tqdm(enumerate(data_dict)):
        data_dict[ts].Crop(min_x, max_x,
                           min_y, max_y,
                           min_z, max_z)


def extract_data_df(data_dict, field_name):
    # Extracts the time series for the specific field at each position

    print("==> Extracting the field \"{}\" from the vtu files.".format(field_name))
    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]

    data_mat = np.zeros((N, T))

    for t, ts in enumerate(data_dict):
        data_mat[:, t] = data_dict[ts].GetField(field_name).T
    return pd.DataFrame(data_mat)


def extract_time_df(data_dict):
    # Extracts the time vector of the simulation
    # Returns a numpy array

    i_0 = list(data_dict.keys())[0]
    T = len(data_dict)
    N = data_dict[i_0].GetLocations().shape[0]

    time_vec = np.zeros((T, 1))

    for t, ts in enumerate(data_dict):
        time_vec[t] = data_dict[ts].GetField('Time')[0]
    return pd.DataFrame(time_vec)


def extract_data_df(data_dict, parameters):
    # Extracts the time series for the specific field at each position
    # Returns a pandas DataFrame

    field_name = parameters["field_name"]

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


def save_data_df(data_df: pd.DataFrame, parameters):
    field_name = parameters["field_name"]

    folder_path = create_temp_folder(parameters)
    file_name = '_'.join(['data', field_name]) + '.pkl'
    full_path = join(folder_path, file_name)
    data_df.to_pickle(full_path)
    print("==> Saving to : {}".format(full_path))


def load_data_df(parameters) -> pd.DataFrame:
    field_name = parameters["field_name"]

    folder_path = create_temp_folder(parameters)
    file_name = '_'.join(['data', field_name]) + '.pkl'
    full_path = join(folder_path, file_name)
    try:
        data_df = pd.read_pickle(full_path)
        print("==> Loading from : {}".format(full_path))
    except:
        print("==> Failed to load : {}".format(full_path))

    return data_df


def save_df(type, data_df: pd.DataFrame, parameters):
    field_name = parameters["field_name"]

    folder_path = create_temp_folder(parameters)
    file_name = '_'.join([type, field_name]) + '.pkl'
    full_path = join(folder_path, file_name)
    data_df.to_pickle(full_path)
    print("==> Saving to : {}".format(full_path))


def load_df(type, parameters) -> pd.DataFrame:
    field_name = parameters["field_name"]

    folder_path = create_temp_folder(parameters)
    file_name = '_'.join([type, field_name]) + '.pkl'
    full_path = join(folder_path, file_name)
    try:
        data_df = pd.read_pickle(full_path)
        print("==> Loading from : {}".format(full_path))
    except:
        print("==> Failed to load : {}".format(full_path))

    return data_df


def create_temp_folder(parameters) -> str:
    i_start = parameters["i_start"]
    i_end = parameters["i_end"]
    crop = parameters["crop"]

    # Creates and Returns the path to the cache folder
    folder_name = '_'.join(['cache', str(i_start), str(i_end), str(crop)])
    full_path = join(data_path, temp_folder, folder_name)
    if not isdir(full_path):
        mkdir(full_path)
    return full_path


# ============================================
# ========== Handling Extracted Files=========
# ============================================

def working_subset(data_df, loc_df, nbins=(25, 25, 25), threshold_sum=10 ** -2):
    """ This function returns the indices of the points in the space which 
        are relevant for our optimisation problem. They are selected by cutting
        the 3D space into cuboids. Those cuboids are of different shape and their number
        is parametrised by nbins in each dimension. In each cuboid, we then compute a spatial
        and temporal sum over the included points. Each cuboid is then kept if the previously 
        summed value is over the specified threshold.
        
    """
    data_df[data_df < 0] = 0
    main_df = loc_df.copy()
    main_df.loc[:, 'data'] = data_df.sum(axis=1)

    main_df.loc[:, 'Xcut'] = pd.cut(main_df.loc[:, 'X'], bins=nbins[0])
    main_df.loc[:, 'Ycut'] = pd.cut(main_df.loc[:, 'Y'], bins=nbins[1])
    main_df.loc[:, 'Zcut'] = pd.cut(main_df.loc[:, 'Z'], bins=nbins[2])

    cut_col = ['Xcut', 'Ycut', 'Zcut']
    windowed_mean_df = main_df.groupby(cut_col).sum().loc[:, ['data']].dropna()
    windowed_mean_df.loc[:, 'indices'] = main_df.groupby(cut_col).apply(lambda x: np.array(x.index.to_list()))
    selected_windows = windowed_mean_df[(windowed_mean_df['data'] > threshold_sum)].dropna()
    working_subset = np.hstack(selected_windows.indices.values)
    #print('The remaining number of points is : ', len(working_subset))
    return working_subset


def human_level_subset(buildingshape, loc_df, h, w):
    """ This function returns a set of points which are taken at h meters from the ground and building tops,
    and w meters from the buildings sides. This is enabled by

    :param buildingshape: the dictionary containing the shapes of the buildings to consider.
    :param loc_df: the pandas dataframe containing the locations
    :param h: the vertical distance from ground/building
    :param w: the horizontal distance from sides of buildings
    :return: the set of points selected
    """
    loc_df.loc[:, "I"] = loc_df.index
    AllPoints = MultiPoint(loc_df.loc[:, ["X", "Y", "I"]].values)

    buildingPoly = dict()
    buildingHeight = dict()
    bPointsSelect = []
    for Nbuild in buildingshape:
        # Building the Polygon of each Building
        buildingPoly[Nbuild] = Polygon(buildingshape[Nbuild])

        # Computing the height of each building :
        # First Select the points which are inside the bounds (-1m buffer to get the inner points only)
        bPoints = AllPoints.intersection(buildingPoly[Nbuild].buffer(-1))
        # Get the index of those points
        idx_bPoints = np.array([p.z for p in bPoints]).astype(int)
        # Height = min of z of those points
        buildingHeight[Nbuild] = loc_df.loc[idx_bPoints, "Z"].min()

        # Select the points which are on top of each building with margin h (top) and w (side)
        bPoints = AllPoints.intersection(buildingPoly[Nbuild].buffer(w))
        idx_bPoints = np.array([p.z for p in bPoints]).astype(int)
        idx_bPointsSelect = loc_df.loc[idx_bPoints, :].loc[loc_df.Z[idx_bPoints] <= buildingHeight[Nbuild] + h,
                            :].index.values
        bPointsSelect.append(list(idx_bPointsSelect))

    # Add the points that are h meters from the ground
    bPointsSelect.append(list(loc_df.loc[loc_df.Z <= h, :].index.values))
    bPointsSelect = np.hstack(bPointsSelect)
    bPointsSelect = list(set(list(bPointsSelect)))

    return bPointsSelect


def set_to_onehot(A, n):
    """Function that maps a list of points to a one hot encoding of selected points"""
    A_list = np.zeros((n, 1))
    A_list[A] = 1
    return A_list


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
