#Title: Homework 8 problem 7 and 8
#Author: Wilfredo Rodezno, Jr.
#Title: Homework 8 problem 7 and 8
#Author: Wilfredo Rodezno, Jr.
import numpy as np
import pandas as pd
from sklearn import svm
import sklearn

#Parameters passed into sklearn svm solver
C = [.0001,.001,.01,.1,1]
q = 2
kernel = 'poly'
gamma = 1
coef0=1

#Initiating variables of interest
SimLen = 100
Num_SupportVectors = []
ListEout = np.zeros(len(C))
ListEin = np.zeros(len(C))
DegPolyEout = {}
DegPolyEin = {}
DegPloySVs = {}
scores = np.zeros(len(C)*SimLen).reshape(SimLen,len(C))
CrossValidationPick = {}
results = {}
winner_value = []


#Downloadning data

TrainData = pd.read_table('trainhw8.txt',header=None, delimiter = "  ",
                          names = ['y','x1','x2'])


TestData = pd.read_table('testhw8.txt',header=None, delimiter = "  ",
                          names = ['y','x1','x2'])


#A : 1 vs 5
#Preparing training data
DataA = TrainData.copy()
DataA = DataA.loc[(DataA.y==1) | (DataA.y==5),['y','x1','x2'] ]
DataA['y'][TrainData['y']==1] = 1
DataA['y'][TrainData['y']== 5] = 0
XA = np.array(DataA.ix[:,1:])
YA = np.array(DataA.ix[:,0])



#Setting the random seed for crossvalidation
np.random.seed(42)
for i in range(SimLen):
    for c in C:
        SolutionA = svm.SVC(C = c,kernel = kernel,degree=q,
                    gamma = gamma, coef0=coef0).fit(XA,YA)    
        #10 fold CrossValidation
        score = sklearn.cross_validation.cross_val_score(SolutionA, XA, YA, cv=10)
        score = 1 - score 
        results[c] = score.mean()

        if c==.001:
            winner_value = [results[c]] + winner_value

    CrossValidationPick[i] = [ans for ans in results.keys() if results[ans]==min(results.values()) ]







        
        
        
        






        
        
        







