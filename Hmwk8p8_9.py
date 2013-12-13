#Title: Homework 8 problem 9 and 10
#Author: Wilfredo Rodezno, Jr.
#Title: Homework 8 problem 9 and 10
#Author: Wilfredo Rodezno, Jr.
import numpy as np
import pandas as pd
from sklearn import svm
import sklearn

#Parameters passed into sklearn svm solver
C = [.01,1,100,10**(4),10**(6)]
degree = [2,5]
kernel = 'rbf'
gamma = 1
coef0=1

#Initiating variables of interest
Num_SupportVectors = []
ListEout = np.zeros(len(C))
ListEin = np.zeros(len(C))
DegPolyEout = {}
DegPolyEin = {}
DegPloySVs = {}




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

#Preparing test data
TESTDataA = TestData.copy()
TESTDataA = TESTDataA.loc[(TESTDataA.y==1) | (TESTDataA.y==5),['y','x1','x2'] ]
TESTDataA['y'][TestData['y']==1] = 1
TESTDataA['y'][TestData['y']== 5] = 0
TESTXA = np.array(TESTDataA.ix[:,1:])
TESTYA = np.array(TESTDataA.ix[:,0])




for i in range(len(C)):
        SolutionA = svm.SVC(C = C[i],kernel = kernel,
                    gamma = gamma, coef0=coef0).fit(XA,YA)
        #Estimating In sample error Ein
        YAestimates = SolutionA.predict(XA)
        A_Accuracy = sklearn.metrics.zero_one_score(YA,YAestimates)
        A_Ein = 1 - A_Accuracy


        #Estimating Out sample error
        TEST_YAestimates = SolutionA.predict(TESTXA)
        TEST_A_Accuracy = sklearn.metrics.zero_one_score(TESTYA,TEST_YAestimates)
        A_Eout = 1 - TEST_A_Accuracy
        

        #Updating Lists
        ListEout[i] = A_Eout
        ListEin[i] = A_Ein
        #Number of support vectors
        Num_SupportVectors = [len(SolutionA.support_vectors_)] + Num_SupportVectors
        







