#Author: Wilfredo Rodezno, Jr.
#Purpose: Caltech Learning from Data Hmwk5 Problem 7
#Performing Stochastic Gradient Descent 

import numpy as np
from numpy import exp
from numpy import dot 
import pandas as pd
import matplotlib.pyplot as plt

#Initating experiment parameters and random seed
exp_length =  100
N=100
d=2  
iterations = np.zeros(exp_length)
Total = 200
E_in = np.zeros(exp_length)
E_out = np.zeros(exp_length)
iterations = np.zeros(exp_length)
nu = .01                                                     #Learning Rate
#Keeping track of the number of epocs before convergence
NumEpochs = np.zeros(exp_length)


#Experiment
for k in range(exp_length):

    #Genterating the target function
    p1 = np.random.uniform(-1,1,d)
    p2 = np.random.uniform(-1,1,d)
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1]-m*p2[0]

    #Evaluating target funtion on training data
    X = np.random.uniform(-1,1,[Total,d])
    Logicals =  X[:,1] < X[:,0]*m+b
    Y = np.zeros(Total)
    Y[Logicals] = -1
    Y[Logicals==False] = 1
    y = Y[:N]

    #Adding a ones column to training data for regression algorithem
    constant = np.ones(Total)
    X = np.column_stack((constant,X))
    #Obtaining the in Sample training data set 
    x = X[:N,:]

    #Initial weights are zero and initial change in Weights
    EpochWeights = []
    w = np.zeros(d+1)
    EpochWeights = EpochWeights + [w]
    ChangeInWeights = 100
    epoch = 0

    #Logistic Regression Stochastic Gradient Descent
    while ChangeInWeights >= .01:   
    #for r in range(800):

        #Shuffling the data after an epoch and keeping track of the number of epochs 
        index = np.array([i for i in range(N)])                        
        np.random.shuffle(index)
        x = x[index,:]
        y = y[index,:]
        

        #Performing Stochastic Gradient Descent on an Epoch

        for i in range(N):
    
            gradient = -(y[i]*x[i,:])/(1 + np.exp(y[i]*np.dot(w,x[i,:])))
            #Updating weights
            w = w - nu*gradient

        
        #Updating the change in weights from one epoch to the next and the number of epochs
        EpochWeights = EpochWeights + [w]
        epoch = epoch + 1
        ChangeInWeights = np.linalg.norm(EpochWeights[epoch-1]-EpochWeights[epoch])
        
    NumEpochs[k] = epoch
    
    #Obtaining the out of sample error for experiment k
    for l in range(len(X[N+1:])):
        E_out[k] = E_out[k] + np.log( 1 +exp( -1.*Y[N+l]*dot(w,X[N+l,:])))
    E_out[k] = (float(1)/len(X[N+1:]))*E_out[k]




#Obtaining results from experiements
OutOfSampleError = np.mean(E_out)

AverageEpochs = np.mean(NumEpochs)
    


















