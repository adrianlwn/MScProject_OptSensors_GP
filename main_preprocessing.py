from utils.data_handling import *
from parameters import *


# Loading the vtu files in a dictionary, along with the location df and the time vec
loaded = load_vtu(i_start, i_end, crop)
data_dict, ref_vtu, loc_df, time_vec = loaded

# Extract the specified field.
data_df = extract_data_df(data_dict, field_name)
print(data_mat.head())

# Save the specified field

data_df.to


# Test Load the specified field






