import numpy as np
from numpy import exp
from numpy import dot
import pandas as pd
import matplotlib.pyplot as plt



#Problem 2
#Loading both training data and test data
#InData is training data
#OutData is test data
InData = pd.read_table('indata.txt', sep = "  ", header = None,names = ['x1','x2','y'])
OutData = pd.read_table('outdata.txt', sep = "  ", header = None,names = ['x1','x2','y'])

#Transforming data 
InData['x1^2'] = InData['x1']**(2)
InData['x2^2'] = InData['x2']**(2)
InData['X1*x2'] = InData['x2']*InData['x1']
InData['|x1 - x2|'] = np.abs(InData['x1'] - InData['x2'])
InData['|x1 + x2|'] = np.abs(InData['x1'] + InData['x2'])
InData['ones'] = np.ones(35)
OutData['x1^2'] = OutData['x1']**(2)
OutData['x2^2'] = OutData['x2']**(2)
OutData['X1*x2'] = OutData['x2']*OutData['x1']
OutData['|x1 - x2|'] = np.abs(OutData['x1'] - OutData['x2'])
OutData['|x1 + x2|'] = np.abs(OutData['x1'] + OutData['x2'])
OutData['ones'] = np.ones(OutData.shape[0])


#Seperating target values and features
features = [n for n in InData.columns if n!='y']
X = InData[:][features]
y = InData['y']

Xout = OutData[:][features]
yout = OutData['y']


#Lambda Constraint
k = [2,1,0,-1,-2]
InSampleError = np.zeros(len(k))
OutSampleError = np.zeros(len(k))


for i in range(len(k)):
    constraint = 10**(k[i])
    #Obtaining linear regression weights from test dataset
    Inverse_Prod = np.linalg.inv(dot(X.transpose(),X)  + constraint*np.eye(len(features)) )
    #Weights
    w = dot(dot(Inverse_Prod,X.transpose()),y)
    #Obtaining In-Sample and out of Sample -Error
    InSampleError[i] = np.sum(np.sign(dot(X,w))!=y )
    InSampleError[i] = float(InSampleError[i])/InData.shape[0]

    OutSampleError[i] = np.sum(np.sign(dot(Xout,w))!=yout )
    OutSampleError[i] = float(OutSampleError[i])/OutData.shape[0]




















