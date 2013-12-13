#Title: Cal Tech Learning From Data Homework 7 Problem 1
#Author: Wilfredo Rodezno

import numpy as np
import pandas as pd
from numpy import dot

InData = pd.read_table('indata.txt', sep = "  ", header=None,
                          names= ['x1','x2','y'])

OutData = pd.read_table('outdata.txt', sep = "  ", header = None,
                         names=['x1','x2','y'])

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


#Splitting InData into a training dataset and a validation dataset  
TrainData = InData[:25]
ValidData = InData[25:]


#Seperating Training target (Ytrain) and training features (Xtrain)
features = [n for n in InData.columns if n!='y']
Xtrain = TrainData[:][features]
Ytrain = TrainData['y']


#Applying linear regresson on the Training set for models k = [3,4,5,6,7]

K = {3:['ones','x1','x2','x1^2'], 4:['ones','x1','x2','x1^2','x2^2'],5:['ones','x1','x2','x1^2','x2^2','X1*x2'], 6:['ones','x1','x2','x1^2','x2^2','X1*x2','|x1 - x2|'],
     7:['ones','x1','x2','x1^2','x2^2','X1*x2','|x1 - x2|', '|x1 + x2|'] }
w = {}


for k in K.keys():
    #Obtaining linear regression weights from Training dataset
    Inverse_Prod = np.linalg.inv(dot(Xtrain[K[k]].transpose(),Xtrain[K[k]]))
    #Weights
    w[k] = dot (dot(Inverse_Prod, Xtrain[K[k]].transpose()) , Ytrain)
    


#Obtaining Validation error using weights obtaining from the training
#Dataset
Xvalid = ValidData[:][features]
Yvalid = ValidData['y']
ValidError = {}
for k in K.keys():
    ValidError[k] = np.sum(np.sign(dot(Xvalid[K[k]],w[k]))!=Yvalid )
    ValidError[k] = float(ValidError[k])/ValidData.shape[0]


#Problem 1 solution
min(ValidError.items(),key = lambda x:x[1])



#Problem 2
#Evaluating the out-of-sample classification error using out.dta on the 5 models 
Xout = OutData[:][features]
Yout = OutData['y']
OutError = {}
for k in K.keys():
    OutError[k] = np.sum(np.sign(dot(Xout[K[k]],w[k]))!=Yout )
    OutError[k] = float(OutError[k])/OutData.shape[0]


#Problem 2 solution
min(OutError.items(),key = lambda x:x[1])




















#Problem 3
#Reverse the role of training and validation set



#Splitting InData into a training dataset and a validation dataset (Reverse)  
ValidData = InData[:25]
TrainData = InData[25:]


#Seperating Training target (Ytrain) and training features (Xtrain)
features = [n for n in InData.columns if n!='y']
Xtrain = TrainData[:][features]
Ytrain = TrainData['y']


#Applying linear regresson on the Training set for models k = [3,4,5,6,7]

K = {3:['ones','x1','x2','x1^2'], 4:['ones','x1','x2','x1^2','x2^2'],5:['ones','x1','x2','x1^2','x2^2','X1*x2'], 6:['ones','x1','x2','x1^2','x2^2','X1*x2','|x1 - x2|'],
     7:['ones','x1','x2','x1^2','x2^2','X1*x2','|x1 - x2|', '|x1 + x2|'] }
w = {}


for k in K.keys():
    #Obtaining linear regression weights from Training dataset
    Inverse_Prod = np.linalg.inv(dot(Xtrain[K[k]].transpose(),Xtrain[K[k]]))
    #Weights
    w[k] = dot (dot(Inverse_Prod, Xtrain[K[k]].transpose()) , Ytrain)
    


#Obtaining Validation error using weights obtaining from the training
#Dataset
Xvalid = ValidData[:][features]
Yvalid = ValidData['y']
ValidError = {}
for k in K.keys():
    ValidError[k] = np.sum(np.sign(dot(Xvalid[K[k]],w[k]))!=Yvalid )
    ValidError[k] = float(ValidError[k])/ValidData.shape[0]


#Problem 3 solution
print min(ValidError.items(),key = lambda x:x[1])



#Problem 4
#Evaluating the out-of-sample classification error using out.dta on the 5 models 
Xout = OutData[:][features]
Yout = OutData['y']
OutError = {}
for k in K.keys():
    OutError[k] = np.sum(np.sign(dot(Xout[K[k]],w[k]))!=Yout )
    OutError[k] = float(OutError[k])/OutData.shape[0]


#Problem 5 solution
print min(OutError.items(),key = lambda x:x[1])































