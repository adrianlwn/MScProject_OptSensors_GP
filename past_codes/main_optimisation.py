from utils.data_handling import *
from utils.sensor_optimisation_gpy import *
from parameters import *

import matplotlib.pyplot as plt

np.random.seed(42)


parameters['i_end'] = 100
print(parameters)


loaded = initial_load_data(parameters, recompute=False)
ref_vtu, data_df, loc_df, time_df = loaded

# Select slice of Data for 2D GP :
I = idx_slice(loc_df, direction='Z', s_min=1, s_max=1.5)


# Data for the regression
dim = 3
t = 100
X = loc_df.values[I,:dim]
Y = data_df.values[I,t].reshape(-1,1)
print(X.shape)


kernel = GPy.kern.Matern52(dim,ARD=True)
print(kernel)

n_V = X.shape[0]
sets = define_sets(n_V)

k = 10
sensor_loc_optimisation_naive(k, sets, X, Y, kernel)

