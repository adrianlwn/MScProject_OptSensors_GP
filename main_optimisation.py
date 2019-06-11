from utils.data_handling import *
from utils.sensor_optimisation import *
from parameters import *


new_data_df = load_data_df(field_name, i_start, i_end, crop)
print(new_data_df.head())