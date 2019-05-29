

def cov_matrix(X):
    ''' Computes the covariance matrix using np.cov function - Not parallel'''
    K = np.cov(X)
    return K


def cov_matrix_p(X, step_size = 1e5, n_processes = 100):
    ''' Computes the covariance matrix in parallel '''
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    v = pd.DataFrame({'index': range(n_samples)})
    
    # index array for parallelization
    pos_array = np.array(np.linspace(0, n_features*(n_features-1)//2, n_processes), dtype=int)
    
    # connector function to compute pairwise pearson correlations
    def cov(index_s, index_t):
        features_s = X[index_s]
        features_t = X[index_t]
        cov = np.einsum('ij,ij->i', features_s, features_t) / n_samples
        return cov

    return K


def mean_matrix(data_mat):
    mean_mat = np.mean(data_mat,axis=1)
    return mean_mat
