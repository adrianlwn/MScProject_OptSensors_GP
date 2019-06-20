from utils.data_handling import *
from utils.covariance import *
from parameters import *

import sklearn.covariance

import matplotlib.pyplot as plt
import seaborn as sns

data_mat = load_data_df(field_name, i_start, i_end, crop).values
print(data_mat)


def plot_matrix(mat, title='Matrix'):
    plt.figure(figsize=(10, 10))
    plt.matshow(mat[:50, :50])
    plt.colorbar()
    plt.title(title)
    plt.show()


sample_cov = np.cov(data_mat)
plot_matrix(sample_cov, 'Sample Covariance')

empirical_cov = sklearn.covariance.empirical_covariance(data_mat, assume_centered=False)
plot_matrix(empirical_cov, 'Empirical Covariance')

graph_cov, graph_prec = sklearn.covariance.graphical_lasso(sample_cov, alpha=0.5)
plot_matrix(graph_prec, 'Graph Covariance a = 0.5')

graph_cov, graph_prec = sklearn.covariance.graphical_lasso(sample_cov, alpha=10)
plot_matrix(graph_prec, 'Graph Covariance a = 10')
