from utils.data_handling import *
from utils.covariance import *
from parameters import *

import sklearn.covariance

import matplotlib.pyplot as plt

data_mat = load_data_df(field_name, i_start, i_end, crop).values
print(data_mat)


def plot_matrix(mat):
    plt.figure(figsize=(10, 10))
    plt.matshow(sample_cov)
    plt.colorbar()
    plt.show()
    print(mat)


sample_cov = np.cov(data_mat)
plot_matrix(sample_cov)


empirical_cov = sklearn.covariance.empirical_covariance(data_mat, assume_centered=False)
plot_matrix(empirical_cov)


graph_cov, graph_prec = sklearn.covariance.graphical_lasso(sample_cov, alpha=0.3)
plot_matrix(graph_cov)
