import pandas as pd
import numpy as np

from scipy.optimize import minimize
import random
import time
import copy
import pdb

from numpy.linalg import inv
from numpy import linalg as LA
import math
from scipy.sparse.linalg import svds
import scipy


import sys
sys.path.append('../fluidity-master/python/')
import vtktools


from scipy.spatial.distance import pdist, squareform


def data_assimilation(indLoc, Z):
    """ Function that proceeds to DA for a limited number of points
        Inputs :
        --- indLoc : indexes of local points for DA
        --- Z : np.array containing all the states and all samples
        Outputs :
        --- MSExb: Background Error
        --- MSExDA: DA Error
    
    """
    ntime = 989
    ntimelocal = 989

    #indLoc = np.sort(indLoc)

    print(indLoc)
    NindLoc = len(indLoc)

    ##

    uvwTot = Z[indLoc,:].T.flatten()
    n = NindLoc

    ##

    dimuvwTot = len(uvwTot)

    m = np.array([])
    m = np.zeros(int(dimuvwTot/ntime))
    for j in range(n):
            for i in range(1,ntime+1):
                    m[j] = np.copy(m[j] + uvwTot[j+(i-1)*n])
            m[j] = m[j]/ntime

    ##


    err = np.array([])
    err = np.zeros(dimuvwTot)

    for j in range(n):
            for i in range(1,ntime+1):
                    err[j+(i-1)*j] = abs(uvwTot[j+(i-1)*j]-m[j])

    W =  np.transpose(np.reshape(err, (ntime,n)))

    ##


    #trnc=n-1
    #Utrunc, strunc, Wtrunc = svds(W, k=trnc)
    #X = Utrunc.dot(np.diag(np.sqrt(strunc)))
    #np.savetxt("./matrixVpreclocal"+str(trnc)+".txt", X)
    #V = X.copy()

    V = W.copy()
    #V = np.loadtxt('Optimal Points/V_oas')

    lam = 1e-60

    ##


    #put the observation file (time step 988)

    #ugg=vtktools.vtu('./WindTunnel/WTobserv/TracerWT400.vtu')
    ugg=vtktools.vtu('../data/small3DLSBU/LSBU_988.vtu')
    ugg.GetFieldNames()
    uvwVecobstot = ugg.GetScalarField('Tracer')
    uvwVecobs = np.array([])
    for i in range(NindLoc):
        indexLocal = indLoc[i]
        indexLocal = int(indexLocal)
        xMpointobs = uvwVecobstot[indexLocal]
        uvwVecobs = np.append(uvwVecobs,xMpointobs)

    #put the background (time step 100)

    nstobs = len(uvwVecobs)
    #ug=vtktools.vtu('./WindTunnel/Projected_Normal_400.vtu')
    ug=vtktools.vtu('../data/small3DLSBU/LSBU_100.vtu')

    ug.GetFieldNames()
    uvwVectot = ug.GetScalarField('Tracer')
    nRec = len(uvwVectot)
    uvwVec = np.array([])
    for i in range(NindLoc):
        indexLocal = indLoc[i]
        indexLocal = int(indexLocal)
        xMpointFl = uvwVectot[indexLocal]
        uvwVec = np.append(uvwVec,xMpointFl)

    nst = len(uvwVec)
    pos=ug.GetLocations()
    z=pos[:,2]
    n = len(uvwVec)

    ##


    m = n #trnc
    xB = uvwVec.copy()
    y = uvwVecobs.copy()
    R = lam * 0.9

    x0 = uvwVec

    Vin = np.linalg.pinv(V)
    v0 = np.dot(Vin,x0)
    VT = np.transpose(V)
    HxB = xB.copy()
    d = np.subtract(y,HxB)



    # Cost function J
    def J(v):
            vT = np.transpose(v)
            vTv = np.dot(vT,v)
            Vv = np.dot(V,v)
            Jmis = np.subtract(Vv,d)
            invR = 1/R
    #       invR = 1e+60
            JmisT = np.transpose(Jmis)
            RJmis = JmisT.copy()
            J1 = invR*np.dot(Jmis,RJmis)
            Jv = (vTv + J1) / 2
            return Jv

    # Gradient of J
    def gradJ(v):
            Vv = np.dot(V,v)
            Jmis = np.subtract(Vv,d)
            invR = 1/R
    #       invR = 1e+60
            g1 = Jmis.copy()
            VT = np.transpose(V)
            g2 = np.dot(VT,g1)
            gg2 = np.multiply(invR , g2)
            ggJ = v + gg2
            return ggJ



    # Compute the minimum
    t = time.time()

    res = minimize(J, v0, method='L-BFGS-B', jac=gradJ,
                    options={'disp': True})


    vDA = np.array([])
    vDA = res.x
    deltaxDA = np.dot(V,vDA)
    xDA = xB + deltaxDA

    elapsed = time.time() - t

    errxB = y - xB
    MSExb = LA.norm(errxB, 2)/LA.norm(y, 2)
    print('L2 norm of the background error' , MSExb )

    errxDA = y - xDA
    MSExDA = LA.norm(errxDA, 2)/LA.norm(y, 2)
    print('L2 norm of the error in DA solution' , MSExDA)

    return MSExb, MSExDA

def data_assimilation_full_set(indLoc, Z):
     """ Function that proceeds to DA for the full set but with a limited number of points observed
        Inputs :
        --- indLoc : indexes of local points for DA
        --- Z : np.array containing all the states and all samples
        Outputs :
        --- MSExb: Background Error
        --- MSExDA: DA Error
    
    """

    ntime = 989

    print(indLoc)
    NindLoc = len(indLoc)
    n = Z.shape[0]


    ##
    uvwTot = Z[:,:].T.flatten()


    ##

    dimuvwTot = len(uvwTot)

    """
    uvwTot = np.array([])
    for i in range(ntime):

        filename = '../data/small3DLSBU/LSBU_'+str(i)+'.vtu'
        ug=vtktools.vtu(filename)
        ug.GetFieldNames()
        uvwVec=ug.GetScalarField('Tracer')
        n=len(uvwVec)
        uvwTot= np.append(uvwTot,uvwVec)



    dimuvwTot = len(uvwTot)
    """
    m = np.array([])
    m = np.zeros(int(dimuvwTot/ntime))
    for j in range(n):
        for i in range(1,ntime+1):
            m[j] = np.copy(m[j] + uvwTot[j+(i-1)*n])
        m[j] = m[j]/ntime



    err = np.array([])
    err = np.zeros(dimuvwTot)

    for j in range(n):
        for i in range(1,ntime+1):
            err[j+(i-1)*j] = abs(uvwTot[j+(i-1)*j]-m[j])

    W =  np.transpose(np.reshape(err, (ntime,n)))

    '''
    mid = int(np.floor(ntime/2))

    U, s, W = svds(V, k=mid)

    print('il primo svd calcolato!')


    st = np.sort(s)[::-1]

    s1= st[0]

    ref=math.sqrt(s1)
    print "ref", ref


    trnc=2

    while st[trnc] >= ref and trnc < mid-1 :
            trnc=trnc+1
            print "trnc", trnc
            print "st[trnc]", st[trnc]

    '''

    trnc = 501


    print('value of trnc')
    print(trnc)


    Utrunc, strunc, Wtrunc = svds(W, k=trnc)
    X = Utrunc.dot(np.diag(np.sqrt(strunc)))
    #np.savetxt("/data/TEST3D/TEST1/MatricesPrec"+str(ntime)+"/matrixVprec"+str(trnc)+".txt", X)


    V = X.copy()



    ugg=vtktools.vtu('../data/small3DLSBU/LSBU_988.vtu')
    ugg.GetFieldNames()
    uvwVecobslocal = ugg.GetScalarField('Tracer')



    ug=vtktools.vtu('../data/small3DLSBU/LSBU_100.vtu')
    ug.GetFieldNames()
    uvwVec = ug.GetScalarField('Tracer')


    uvwVecobs = uvwVec.copy()
    uvwVecobs[indLoc] = uvwVecobslocal[indLoc]




    pos=ug.GetLocations()
    z=pos[:,2]

    lam = 0.1e-60

    n = len(uvwVec)
    m = trnc


    # Observations in some points
    xB = uvwVec.copy()


    y = uvwVecobs.copy()





    R = lam * 0.9

    x0 = np.ones(n)

    ##########V = np.loadtxt('/homes/rarcucci/4DVAR-ROX/VarDACode/small3DCase/matrixV'+str(m)+'-velocity.txt', usecols=range(m))




    Vin = np.linalg.pinv(V)


    v0 = np.dot(Vin,x0)




    VT = np.transpose(V)
    HxB = xB.copy()
    d = np.subtract(y,HxB)


    # Cost function J

    def J(v):
        vT = np.transpose(v)
        vTv = np.dot(vT,v)
        Vv = np.dot(V,v)
        Jmis = np.subtract(Vv,d)
        invR = 1/R
        JmisT = np.transpose(Jmis)
        RJmis = JmisT.copy()
        J1 = invR*np.dot(Jmis,RJmis)
        Jv = (vTv + J1) / 2
        return Jv


    # Gradient of J

    def gradJ(v):
        Vv = np.dot(V,v)
        Jmis = np.subtract(Vv,d)
        invR = 1/R
        g1 = Jmis.copy()
        VT = np.transpose(V)
        g2 = np.dot(VT,g1)
        gg2 = np.multiply(invR , g2)
        ggJ = v + gg2
        return ggJ



    # Compute the minimum


    t = time.time()

    res = minimize(J, v0, method='L-BFGS-B', jac=gradJ,
                    options={'disp': True})


    vDA = np.array([])
    vDA = res.x
    deltaxDA = np.dot(V,vDA)
    xDA = xB + deltaxDA

    elapsed = time.time() - t



    errxB = y - xB
    MSExb = LA.norm(errxB, 2)/LA.norm(y, 2)
    print('L2 norm of the background error' , MSExb )

    errxDA = y - xDA
    MSExDA = LA.norm(errxDA, 2)/LA.norm(y, 2)
    print('L2 norm of the error in DA solution' , MSExDA)

    return MSExb, MSExDA



def ball_set(A,loc_df,radius):
     """ Function that computes the selection of points in a ball arround each element of A, within a radius of "radius"
        Inputs :
        --- A: set of indexes of points
        --- loc_df: pandas dataframe containing the locations of every point of the space
        --- radius: radius of the selection ball
        Outputs :
        --- A_ball: dictionnary indexed by A and containing lists of indexed points of each ball.
    
    """
    A_ball = dict()
    for i,y in enumerate(A):
        print(i,y)
        dist_to_y = (loc_df - loc_df.loc[y,:]).apply(np.linalg.norm,axis=1).drop(y,axis=0)
        A_ball[y] = dist_to_y.loc[dist_to_y < radius].index.tolist()

    return A_ball

def randomise_set(A_ball):
     """ Function that selects randomly a point in each ball arround the set of points A. (see function ball_set)
        Inputs :
        --- A_ball: dictionnary indexed by A and containing lists of indexed points of each ball.
        Outputs :
        --- A_rand: list of indexes close the the original A. 
    
    """
    A_rand = []
    for y in A_ball:
        y_rand = np.random.choice(A_ball[y])
        A_rand.append(y_rand)

    return A_rand