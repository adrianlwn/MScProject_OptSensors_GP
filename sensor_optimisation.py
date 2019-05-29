
def define_sets(n_S = None,seed = 42):
    """ Function that defines the sets used in the optimsation problem. S is randomly selected in V. 
        Inputs : 
        --- n_S : number of sensor to randomly selected in V (if None, S = V ) 
        --- seed : seed used for the random selection of S
        Outputs : 
        --- n_V, V : size and idx of points contained in V : set of all points
        --- n_S, S : size and idx of points contained in S : set of potential sensor points
        --- n_U, U : size and idx of points contained in U : set of non sensor points
        
    """
    n_V = location_df.shape[0]
    V = np.array(range(n_V))

    if n_S == None : 
        n_S = n_V
        S = V
    else :
        np.random.seed(seed)
        S = np.random.choice(V, n_S, replace=False,)
    
    n_U = n_V - n_S
    U = np.setdiff1d(V, S, assume_unique=True)
    
    return n_V, V, n_S, S, n_U, U

def approx_max_info(k):
    """ This function implements the Algorithm 1: Approximation algorithm 
    for maximizing mutual information.
    
    Input Arguments : 
    --- k : number of Sensors to place in S ( k <= n_S )
    
    Needed global variables : 
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points
    
    """
    
    n_A = 0
    A = np.array([])
    
    # Main Loop of the Algorithm : Iterating over the number of sensors to place
    
    for j in tqdm.tqdm_notebook(range(k)):
        
        S_A = np.setdiff1d(S,A).astype(int)
        delta_y = np.zeros((n_S - j))
        ## Inner Loop : Iterating over the potential sensor places
        for i,y in enumerate(S_A):

            A_ = np.setdiff1d(V,np.append(A,[y]))
            # Mutual Information Gain

            delta_y[i] = H_cond(y, A) / H_cond(y, A_)
        
        # Greedily selection the best point to add to A
        y_opt = S_A[np.argmax(delta_y)]
        print((delta_y >0).sum()/len(delta_y))
        # Add the selected point to A
        n_A += 1
        A = np.append(A,y_opt).astype(int)
        
    return A

def approx_lazy_max_info(k):
    """ This function implements the Algorithm 2: Approximation algorithm for 
    maximizing mutual information efficiently using lazy evaluation.
    
    Input Arguments : 
    --- k : number of Sensors to place
    
    Needed global variables : 
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points
    
    """

    # INIT : 
    n_A = 0
    A = np.array([])
    
    delta_y = -1*np.inf*np.ones((n_S,1))
    counter_y = -1*np.ones((n_S,1))
    
    # Each Node of the Heap contains a tupple : (-delta_y, index of point, count_y)
    delta_heap = list(zip(delta_y, S, counter_y))
    heapq.heapify(delta_heap)
    
    # MAIN LOOP of the Algorithm : Iterating over the number of sensors to place
    for j in tqdm.tqdm_notebook(range(k)):
                
        ## INNER LOOP : Iterating over the potential sensor places
        while True : 
            delta, y_opt, count = heapq.heappop(delta_heap)
            if count == j :
                break
            else : 
                A_ = np.setdiff1d(V,np.append(A,[y_opt]))
                # Mutual Information Gain
                delta_y_opt = H_cond(y_opt, A) / H_cond(y_opt, A_)
                heapq.heappush(delta_heap, (-1*delta_y_opt , y_opt ,j) )
        
        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)       
    return A

def approx_local_max_info(k, epsilon):
    """ NOT FININISHED : This function implements the Algorithm 3: Approximation algorithm for 
    maximizing mutual information efficiently using local kernels.
    
    Input Arguments : 
    --- k : number of Sensors to place
    --- epsilon : threshold for local kernel
    
    Needed global variables : 
    --- K : Covariance Matrix between all points
    --- V : Set of all points
    --- S : Set of potential sensor points
    --- n_S : number of such points
    
    """

    # INIT : 
    n_A = 0
    A = np.array([])
    
    delta_y = -1*np.inf*np.ones((n_S,1))
    
    for i,y in enumerate(S):
        delta_y[i] = -1*delta_MI(y,A)
    
    counter_y = -1*np.ones((n_S,1))
    
    # Each Node of the Heap contains a tupple : (-delta_y, index of point, count_y)
    delta_heap = list(zip(delta_y, S, counter_y))
    heapq.heapify(delta_heap)
    
    # MAIN LOOP of the Algorithm : Iterating over the number of sensors to place
    for j in tqdm.tqdm_notebook(range(k)):
        
        delta, y_opt, count = heapq.heappop(delta_heap)
        
        # Add the selected point to A
        n_A += 1
        A = np.append(A, y_opt).astype(int)
        A_ = np.setdiff1d(V,np.append(A,[y]))
        
        loc_A = local_set(y, A, epsilon) 
        loc_A_ = local_set(y, A_, epsilon) 
        
        ## INNER LOOP : Iterating over the potential sensor places
        for i in A: 

            # Mutual Information Gain
            delta_y = H_cond(y, loc_A) / H_cond(y, loc_A_)

            heapq.heappush(delta_heap,(-1*delta_y ,y ,j))
       
    return A
        

def delta_MI(y,A):
    """ Function that computes the Mutual Entropy Difference """
    A_ = np.setdiff1d(V,np.append(A,[y]))
    
    return H_cond(y, A) / H_cond(y, A_)
    
def H_cond(y,X):
        """ Function that returns the conditional Entropy of y knowing X """        
        return K[y,y] - K[np.ix_([y],X)] @ np.linalg.inv(K[np.ix_(X,X)]) @ K[np.ix_(X,[y])] 
        #return K[y,y] - K[np.ix_([y],X)] @ np.linalg.solve(K[np.ix_(X,X)], K[np.ix_(X,[y])])

def local_set(y, X, epsilon):
    """ Function that returns a set of points X_trunc for which K[y,X_trunc] > epsilon
        X being the input set
        Implementing the idea of local Kernels.
        
    """
    i_trunc = (np.abs( K[np.ix_([y],X)] ) > epsilon).flatten()
    X_trunc = X[i_trunc]    
    
    return X_trunc
