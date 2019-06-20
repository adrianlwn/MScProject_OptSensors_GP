from utils.data_handling import *
from parameters import *

# Loading the vtu files in a dictionary, along with the location df and the time vec
print(parameters)
loaded = load_vtu(parameters)
data_dict, ref_vtu, loc_df, time_vec = loaded

# Extract the specified field.
data_df = extract_data_df(data_dict, parameters)
print(data_df.head())

# Save the specified field
save_data_df(data_df, parameters)

# Test Load the specified field

new_data_df = load_data_df(parameters)
print(new_data_df.head())
