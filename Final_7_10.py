import numpy as np
import pandas as pd
from numpy import dot



TrainData = pd.read_table('trainhw8.txt',header=None, delimiter = "  ",
                          names = ['y','x1','x2'])
TrainData['ones'] = np.ones(TrainData.shape[0])

TestData = pd.read_table('testhw8.txt',header=None, delimiter = "  ",
                          names = ['y','x1','x2'])
TestData['ones'] = np.ones(TestData.shape[0])

alpha = 1
features = [n for n in TrainData.columns if n!='y']


##############################################################################################
#Transformed Data
zTrainData = TrainData.copy()

zTrainData['x1^2'] = zTrainData['x1']**(2)
zTrainData['x2^2'] = zTrainData['x2']**(2)
zTrainData['X1*x2'] = zTrainData['x2']*zTrainData['x1']

zTestData = TestData
zTestData['x1^2'] = zTestData['x1']**(2)
zTestData['x2^2'] = zTestData['x2']**(2)
zTestData['X1*x2'] = zTestData['x2']*zTestData['x1']
zfeatures = [n for n in zTrainData.columns if n!='y']

S = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
zOutSampleError = {}
zInSampleError = {}

for s in S:
    Data = zTrainData.copy()
    Data['y'][zTrainData['y']==S[s]] = 1
    Data['y'][zTrainData['y']!= S[s]] = -1
    X = np.array(Data.ix[:,1:])
    Y = np.array(Data.ix[:,0])
    #Obtaining linear regression weights from test dataset
    Inverse_Prod = np.linalg.inv(dot(X.transpose(),X)  + alpha*np.eye(len(zfeatures)) )
    #Weights
    w = dot(dot(Inverse_Prod,X.transpose()),Y)
    #Obtaining In-Sample and out of Sample -Error
    zInSampleError[S[s]] = np.sum(np.sign(dot(X,w))!=Y)
    zInSampleError[S[s]] = float(zInSampleError[S[s]])/zTrainData.shape[0]
    Xout = zTestData[:][zfeatures].copy()
    yout = zTestData['y'].copy()
    yout[zTestData['y']!=S[s]] = -1
    yout[zTestData['y']==S[s]] = 1

    zOutSampleError[S[s]] = np.sum(np.sign(dot(Xout,w))!=yout )
    zOutSampleError[S[s]] = float(zOutSampleError[S[s]])/zTestData.shape[0]





#############################################################################################################################################
#For Transformed Data  

S = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
OutSampleError = {}
InSampleError = {}

for s in S:
    Data = TrainData.copy()
    Data['y'][TrainData['y']==S[s]] = 1
    Data['y'][TrainData['y']!= S[s]] = -1
    X = np.array(Data.ix[:,1:])
    Y = np.array(Data.ix[:,0])
    #Obtaining linear regression weights from test dataset
    Inverse_Prod = np.linalg.inv(dot(X.transpose(),X)  + alpha*np.eye(len(features)) )
    #Weights
    w = dot(dot(Inverse_Prod,X.transpose()),Y)
    #Obtaining In-Sample and out of Sample -Error
    InSampleError[S[s]] = np.sum(np.sign(dot(X,w))!=Y)
    InSampleError[S[s]] = float(InSampleError[S[s]])/TrainData.shape[0]
    Xout = TestData[:][features].copy()
    yout = TestData['y'].copy()
    yout[TestData['y']!=S[s]] = -1
    yout[TestData['y']==S[s]] = 1

    OutSampleError[S[s]] = np.sum(np.sign(dot(Xout,w))!=yout )
    OutSampleError[S[s]] = float(OutSampleError[S[s]])/TestData.shape[0]



#######################################################
#Problem 10 Training the one vs 5 classifier 


zOutSampleError = {}
zInSampleError = {}
Alpha = {'A':.01,'B':1}
for a in Alpha:
    Data = zTrainData.copy()
    Data = Data.loc[(Data.y==1) | (Data.y==5),:  ]
    Data['y'][zTrainData['y']==1] = 1
    Data['y'][zTrainData['y']== 5] = -1
    X = np.array(Data.ix[:,1:])
    Y = np.array(Data.ix[:,0])
    #Obtaining linear regression weights from test dataset
    Inverse_Prod = np.linalg.inv(dot(X.transpose(),X)  + Alpha[a]*np.eye(len(zfeatures)) )
    #Weights
    w = dot(dot(Inverse_Prod,X.transpose()),Y)
    #Obtaining In-Sample and out of Sample -Error
    zInSampleError[Alpha[a]] = np.sum(np.sign(dot(X,w))!=Y)
    zInSampleError[Alpha[a]] = float(zInSampleError[Alpha[a]])/Y.shape[0]

    OutData = zTestData.loc[(zTestData.y==1) | (zTestData.y==5),:].copy()
    OutData['y'][zTrainData['y']==1] = 1
    OutData['y'][zTrainData['y']== 5] = -1
    Xout = np.array(OutData.ix[:,1:])
    yout = np.array(OutData.ix[:,0])
    
    zOutSampleError[Alpha[a]] = np.sum(np.sign(dot(Xout,w))!=yout )
    zOutSampleError[Alpha[a]] = float(zOutSampleError[Alpha[a]])/yout.shape[0]


print zOutSampleError

print zInSampleError













