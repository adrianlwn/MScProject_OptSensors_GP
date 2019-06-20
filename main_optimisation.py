from utils.data_handling import *
from utils.sensor_optimisation import *
from parameters import *


new_data_df = load_data_df(parameters)
print(new_data_df.head())