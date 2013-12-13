#Title: Final Problem 12 December 2013
#Author: Wilfredo Rodezno, Jr.
import numpy as np 
import cvxopt
from numpy  import dot
from numpy import sqrt
from numpy.linalg import inv
from sklearn import svm
from cvxopt import matrix
from cvxopt import solvers



N=7
#Creating Training Data
x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])
y = y.reshape((len(y),1))

K_x_xT = dot(1 + dot(x,x.transpose()),(1 + dot(x,x.transpose())).transpose())     




#Upper bound for quadratic programming solver
UpperBound = 100000000

#Support Vector Machines
#using cvxopt.solver.qp    
y = matrix(y)
K_x_xT = matrix(K_x_xT, tc = 'd')
Py = y*y.T
P = cvxopt.mul(K_x_xT,Py)
P = matrix(P, tc='d')
q = cvxopt.matrix(-1*np.ones(N).reshape((N,1)))
G = cvxopt.matrix(np.diag(np.ones(N) * -1))
h = cvxopt.matrix(np.zeros(N))   
A = y.T
A = matrix(A, tc='d')
b = cvxopt.matrix(0.0)
solution = cvxopt.solvers.qp(P,q,G,h,A,b)

#Obtaining the Lagrange multipliers 
a = np.ravel(solution['x']).reshape((N,1))

#Obtaining the SVM weights
y = np.ravel(y).reshape(N,1)
x = np.ravel(x).reshape(N,2)
w = dot((a*y).transpose(),x)


#Hunting done those support vectors 
index_sv = [k for k in range(len(a)) if a[k] > .3]
support_vectors = a[a>.00001]
#Finding the bias term 
b = y[index_sv[0]] - dot(w,x[index_sv[0]])
    
    




    
    


    

            


    




