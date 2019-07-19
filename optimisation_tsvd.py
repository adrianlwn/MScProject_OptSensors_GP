
from utils.data_handling import *
from utils.sensor_optimisation_tsvd import *
from parameters import *
import scipy.stats

import matplotlib.pyplot as plt
np.random.seed(101)


# ### Importing the Tracer Data to Select the Optimisation set S
print("=========> Importation of the Tracer Data")
parameters['i_end'] = 988
parameters['field_name'] = "Tracer"
print(parameters)

print(" ")
loaded = initial_load_data(parameters, recompute=False)
ref_vtu, data_df, loc_df, time_df = loaded


### Working subset of the data : set S

print("=========> Creation of the working subset")
S = working_subset(data_df, loc_df, nbins = (25,25,25), threshold_sum = 10**-2 )


X = loc_df.values[:,:]
Z = data_df.values[:,:]


# ### Detrending Data :

Z[S,:] = (Z[S,:]  - Z[S,:].mean(axis=1,keepdims=True))


# ### Sensor Optimisation with TSVD :

# Define the Sets for the optimisation

sets = define_sets(S)


# Number of sensors to place :
k = 5

# Truncation parameteter for the TSVD :
tau = 50


A_opt = {}



A_opt['lazy'] = sensor_loc_optimisation_lazy(k,Z, sets, tau)


print("The Optimal set of points is :")
print(A_opt['lazy'])

A = set_to_onehot(A_opt['lazy'],len(Z))

save_vtu(ref_vtu[0],"A_lazy_k5",A)
