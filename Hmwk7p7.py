#Title: Homework 7 Problem 7
#Author: Wilfredo Rodezno


import numpy as np 
import pandas as pd
from numpy  import dot
from numpy import sqrt
from numpy.linalg import inv

#Possible values for p
P = {'a':sqrt(sqrt(3)+4),'b':sqrt(sqrt(3)-1),'c': sqrt(9+4*sqrt(6)),
     'd':sqrt(9 - sqrt(6)) }
#Variable that will compare models for the different values of P
Compare_Models = {}

for p in P:
    X = np.array([-1,P[p],1]).reshape((3,1))
    Y = np.array([0,1,0]).reshape((3,1))
    LinError = []
    ConstError = []
    #One Leave out Cross-validation
    for i in range(len(X)):
        #Linear Model
        Xcross = np.delete(X,i,0)
        Ycross = np.delete(Y,i)
        m = float((Ycross[0] - Ycross[1]))/(Xcross[0] - Xcross[1])
        b = Ycross[0] - m*Xcross[0]
        LinError = LinError + [((X[i]*m + b) - Y[i])**(2)]
        #Constant Model
        ConstError = ConstError + [(np.mean(Ycross) -  Y[i])**(2)]

    Cross_LinError = np.mean(np.array(LinError))
    Cross_ConstError = np.mean(np.array(ConstError))
    Compare_Models[p] = np.abs(Cross_LinError - Cross_ConstError)
        
        




        
