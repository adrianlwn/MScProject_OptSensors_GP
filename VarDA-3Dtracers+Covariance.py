import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import time

from numpy.linalg import inv
from numpy import linalg as LA


import math
from scipy.sparse.linalg import svds


import sys
sys.path.append('../fluidity-master/python')
import vtktools


ntime = 988

uvwTot = np.array([])
for i in range(ntime):

    filename = '../data/small3DLSBU/LSBU_'+str(i)+'.vtu'
    ug=vtktools.vtu(filename)
    ug.GetFieldNames()
    uvwVec=ug.GetScalarField('Tracer')
    n=len(uvwVec)
    uvwTot= np.append(uvwTot,uvwVec)



dimuvwTot = len(uvwTot)

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

V =  np.transpose(np.reshape(err, (ntime,n)))

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


Utrunc, strunc, Wtrunc = svds(V, k=trnc)
X = Utrunc.dot(np.diag(np.sqrt(strunc)))
#np.savetxt("/data/TEST3D/TEST1/MatricesPrec"+str(ntime)+"/matrixVprec"+str(trnc)+".txt", X)


V = X.copy()



ugg=vtktools.vtu('../data/small3DLSBU/LSBU_988.vtu')
ugg.GetFieldNames()
uvwVecobslocal = ugg.GetScalarField('Tracer')



ug=vtktools.vtu('../data/small3DLSBU/LSBU_100.vtu')
ug.GetFieldNames()
uvwVec = ug.GetScalarField('Tracer')

indLoc = np.loadtxt('Optimal Points/localpoints_optimal_eps10-6.txt')
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
