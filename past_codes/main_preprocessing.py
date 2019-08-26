from utils.data_handling import *
from parameters import *

# Loading the vtu files in a dictionary, along with the location df and the time vec
print(parameters)
loaded = initial_load_data(parameters, reload=True)
ref_vtu, data_df, loc_df, time_df = loaded

print(ref_vtu)
