#Title: Homework 8 problem 3
#Author: Wilfredo Rodezno, Jr.
#Title: Homework 8 problem 2
#Author: Wilfredo Rodezno, Jr.
import numpy as np
import pandas as pd
from sklearn import svm
import sklearn

#Parameters passed into sklearn svm solver
C = .01
degree = 2
kernel = 'poly'
gamma = 1
coef0=1

#Downloadning data

TrainData = pd.read_table('trainhw8.txt',header=None, delimiter = "  ",
                          names = ['y','x1','x2'])


#A : 1 vs all
DataA = TrainData.copy()
DataA['y'][TrainData['y']==1] = 1
DataA['y'][TrainData['y']!= 1] = 0
XA = np.array(DataA.ix[:,1:])
YA = np.array(DataA.ix[:,0])
SolutionA = svm.SVC(C = C,kernel = kernel,degree=2,
                    gamma = gamma, coef0=coef0).fit(XA,YA)
#Estimating In sample error Ein
YAestimates = SolutionA.predict(XA)
A_Accuracy = sklearn.metrics.zero_one_score(YA,YAestimates)
A_Ein = 1 - A_Accuracy






#B : 3 vs all
DataB = TrainData.copy()
DataB['y'][TrainData['y']==3] = 1
DataB['y'][TrainData['y']!= 3] = 0
XB = np.array(DataB.ix[:,1:])
YB = np.array(DataB.ix[:,0])
SolutionB = svm.SVC(C = C,kernel = kernel,degree=2,
                    gamma = gamma, coef0=coef0).fit(XB,YB)
#Estimating In sample error Ein
YBestimates = SolutionB.predict(XB)
B_Accuracy = sklearn.metrics.zero_one_score(YB,YBestimates)
B_Ein = 1 - B_Accuracy







#C : 5 vs all
DataC = TrainData.copy()
DataC['y'][TrainData['y']==5] = 1
DataC['y'][TrainData['y']!= 5] = 0
XC = np.array(DataC.ix[:,1:])
YC = np.array(DataC.ix[:,0])
SolutionC = svm.SVC(C = C,kernel = kernel,degree=2,
                    gamma = gamma, coef0=coef0).fit(XC,YC)
#Estimating In sample error Ein
YCestimates = SolutionC.predict(XC)
C_Accuracy = sklearn.metrics.zero_one_score(YC,YCestimates)
C_Ein = 1 - C_Accuracy






#D : 7 vs all
DataD = TrainData.copy()
DataD['y'][TrainData['y']==7] = 1
DataD['y'][TrainData['y']!= 7] = 0
XD = np.array(DataD.ix[:,1:])
YD = np.array(DataD.ix[:,0])
SolutionD = svm.SVC(C = C,kernel = kernel,degree=2,
                    gamma = gamma, coef0=coef0).fit(XD,YD)
#Estimating In sample error Ein
YDestimates = SolutionD.predict(XD)
D_Accuracy = sklearn.metrics.zero_one_score(YD,YDestimates)
D_Ein = 1 - D_Accuracy






#E : 9 vs all
DataE = TrainData.copy()
DataE['y'][TrainData['y']==9] = 1
DataE['y'][TrainData['y']!= 9] = 0
XE = np.array(DataE.ix[:,1:])
YE = np.array(DataE.ix[:,0])
SolutionE = svm.SVC(C = C,kernel = kernel,degree=2,
                    gamma = gamma, coef0=coef0).fit(XE,YE)
#Estimating In sample error Ein
YEestimates = SolutionE.predict(XE)
E_Accuracy = sklearn.metrics.zero_one_score(YE,YEestimates)
E_Ein = 1 - E_Accuracy

print A_Ein
print B_Ein
print C_Ein
print D_Ein
print E_Ein


#Number of support vectors
SolutionA.support_
