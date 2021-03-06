#Author: Wilfredo Rodezno
#Purpose: CalTech Learning From Data, Homework 2 Problem5
#Data: 10/9/2013

import numpy as np

#Initating experiment parameters and random seed
#np.random.seed(0);
exp_length =  1000
N=100 
d=2  
iterations = np.zeros(exp_length)
Total = 1000+1000
E_in = np.zeros(exp_length)
E_out = np.zeros(exp_length)



for i in range(exp_length):

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
    x = X[:N,:]


            
    #Obtaining Weights of Linear Regression using Least Squares w = inv(X'X)*X'y
    w =   np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)


    #Obtaining the Hypothesis's Values from Linear Regression
    g = np.sign(np.dot(x,w))


    #In-Sample Performance  
    E_in[i] = np.sum(g!=y,dtype='float')/(N)

    #Out-Of-Sample Performance
    g = np.sign(np.dot(X[N:,:],w))
    E_out[i] = np.sum(g!=Y[N:],dtype='float')/(Total-N)


    
    


    
print(np.average(E_out))



    
    

            
        





