#Author: Wilfredo Rodezno
#Purpose: CalTech Learning From Data, Homework 2 Problem8
#Data: 10/9/2013

import numpy as np

#Initial variables
N=1000
Sim_Length = 1000
E_in = np.zeros(Sim_Length)
Total = 1000 +1000
d = 2
Subset_Size = .1*Total




for i in range(Sim_Length):
    #Generating Training dataset
    #Evaluating Target Function on dataset
    #Adding Noise to Target function

    X = np.random.uniform(-1,1,[Total,2])
    Y = np.sign( X[:,0]**(2) + X[:,1]**(2) - .6)
    Subset = np.random.randint(Total,size = Subset_Size)
    Y[Subset] = -1*Y[Subset]

    #Obtaining Training dataset and preparing for regression
    constant = np.ones(Total)
    X = np.column_stack((constant,X))
    x = X[:N,:]
    y = Y[:N]


    #Obtaining Weights of Linear Regression using Least Squares w = inv(X'X)*X'y
    #(Initial Weights for Perceptron)
    w =   np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)


    #Obtaining the Hypothesis's Values from Linear Regression  weights 
    g = np.sign(np.dot(x,w))


    #In-Sample Performance  
    E_in[i] = np.sum(g!=y,dtype='float')/(N)


print(np.average(E_in))








