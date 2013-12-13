#Title: Homework 7 Problem 8
#Author: Wilfredo Rodezno, Jr.
import numpy as np 
import cvxopt
from numpy  import dot
from numpy import sqrt
from numpy.linalg import inv
from sklearn import svm
from cvxopt import matrix
from cvxopt import solvers




#Initating experiment parameters and random seed
np.random.seed(0)
exp_length = 1000 
N=100 
d=2  
iterations = np.zeros(exp_length)
Total = 100000
PLA_Eout = np.zeros(exp_length)
SVM_Eout = np.zeros(exp_length)
Num_SupportVectors = np.zeros(exp_length)

#Upper bound for quadratic programming solver
UpperBound = 100000000


for i in range(exp_length):

    #Genterating the target function
    p1 = np.random.uniform(-1,1,d)
    p2 = np.random.uniform(-1,1,d)
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1]-m*p2[0]

    #Evaluating target funtion on training data.
    #Also, Making sure not all target variables,y, equal 1
    #or -1
    y = np.zeros(N)
    while sum(y==1)==0 or sum(y==-1)==0:
        X = np.random.uniform(-1,1,[Total,d])
        Logicals =  X[:,1] < X[:,0]*m+b
        Y = np.zeros(Total)
        Y[Logicals] = -1
        Y[Logicals==False] = 1
        y = Y[:N]
    
    

    #Adding a ones column to training data for perceptron algorithem
    constant = np.ones(Total)
    XPLA = np.column_stack((constant,X))
    xPLA = XPLA[:N,:]
    #x used for support vector machines 
    x = X[:N,:]
    

            
    #Initial weights of perceptron and hypothesis
    w = np.array([0,0,0])
    iteration = 1
    h = np.sign(np.dot(xPLA,w))
    #Performing Perceptron algorithm
    while np.all(y==h)==False:
        j = np.random.randint(0,len(h))
        if h[j]!=y[j]:
            w = w + y[j]*xPLA[j,:]
            iteration = iteration + 1
            h = np.sign(np.dot(xPLA,w))

    iterations[i] = iteration
    
    #Accuracy of out of sample PLA
    
    gPLA = np.sign(np.dot(XPLA[N:,:],w))
    PLA_Eout[i] = np.sum(gPLA!=Y[N:],dtype='float')/(Total-N)
##########################################################################
    #Support Vector Machines
    #using cvxopt.solver.qp
    y = y.reshape((N,1))
    y = matrix(y)
    x = matrix(x)
    Px = x*x.T
    Py = y*y.T
    P = cvxopt.mul(Px,Py)
    q = cvxopt.matrix(-1*np.ones(N).reshape((N,1)))
    G = cvxopt.matrix(np.diag(np.ones(N) * -1))
    h = cvxopt.matrix(np.zeros(N))   
    A = y.T
    b = cvxopt.matrix(0.0)
    solution = cvxopt.solvers.qp(P,q,G,h,A,b)
    #Obtaining the Lagrange multipliers 
    a = np.ravel(solution['x']).reshape((N,1))
    #Obtaining the SVM weights
    y = np.ravel(y).reshape(N,1)
    x = np.ravel(x).reshape(N,2)
    w = dot((a*y).transpose(),x)
    #Hunting done those support vectors 
    index_sv = [k for k in range(len(a)) if a[k] > .0001]
    support_vectors = a[a>.00001]
    #Finding the bias term 
    b = y[index_sv[0]] - dot(w,x[index_sv[0]])
    #Accuracy of out of sample SVM
    gSVM = np.sign(np.dot(X[N:,:],w.transpose()) + b)
    
    SVM_Eout[i] = np.sum(gSVM!=Y[N:].reshape((len(Y[N:]),1)),dtype='float')/(Total-N)
    #Recording the number of support vectors
    Num_SupportVectors[i] = len(index_sv)    
    
    
    
    




#Results from experiment
Avg_SVM_Eout = np.average(SVM_Eout) 
Avg_PLA_Eout = np.average(PLA_Eout)

Percent_SVM_better_PLA = 100*(np.sum(SVM_Eout < PLA_Eout,dtype='float')/exp_length)
Avg_Num_SupportVectors = np.average(Num_SupportVectors)
