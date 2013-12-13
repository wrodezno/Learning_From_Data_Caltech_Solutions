#Author: Wilfredo Rodezno
#Purpose: CalTech Learning From Data, Final Problem 13_15
#Data: 12/5/2013

import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.linalg import inv
import sklearn as sk
from sklearn.svm import SVC



######################################################################################################################################################################
#Making dictionaries that will be used to obtain RBF



#Making a Dictionary that will compute Q matrix
#consisting of elements exp(-rho*||X - mu||^2)
#See Lecture 16 slide 14 for matrix
def Qmatrix(centers,K,data,rangeObs,rho):
    """
    Inputs: centers is a dictionary that maps center labels to center values obtained from Lloyds algorithm
            K is an interger number indicating the number of centers to be used in Lloyds algorithm
            data is the data the was clustered in Lloyds algorithm. It is an n-dimensional numpy array
            rangeObs is the range in form of a list of obs from the data to be used in Lolyds Algorithm
            rho is the coeffient in rbf.
            Note the index names for centers must be of the form jmu where j is some integer

    Output: Q matrix consisting of elements exp(-rho*||X - mu||^2). See Lecture 16 slide 14 for matrix
            
    """
    Qcolumns = {str(i)+'mu':[] for i in range(K)}
    for mu in centers:
        for i in range(rangeObs[0],rangeObs[1]):
            Qcolumns[mu].append( norm(data[i,:] - centers[mu]) )
    Q = np.array([Qcolumns[mu] for mu in centers]).transpose()
    Q = (-1)*rho*Q
    Q = np.exp(Q)
    Q = np.hstack((Q,np.ones(Q.shape[0]).reshape(Q.shape[0],1)))
    #Finally we get Q!!!!!!!!!!
    
    return Q



#Making dictionary that will compute k-centers using Lloyds Algorithm
def Centers(data,K):
    """
    Inputs: data is an N-dimensional array representing data that will be clustered by K-means using Lloyds algorithm
            K is the number of clusters that will be used in K-means clustering

    Output: K-mean centers resulting from Lloyds algorithm
    """

    #Initial values for the centers
    OldMu = np.random.uniform(-1,1,(K,2))
    NewMu = np.zeros((K,2))
    #Distance between centers from different iterations
    DistMu = 100000 
    while DistMu > .0001:
    
        #Dictionary that will keep track of the distance 
        #between centers and data poitns in K-means clustering 
        dis = {}

        #Initiating Clusters that will be formed by K-means clustering
        Clusters = {i:[] for i in range(K)}

     
        for i in range(data.shape[0]):
            for mu in range(K):
                dis[mu] = norm(data[i,:] - OldMu[mu])

            #Finding center with minimum distrance to the point
            MinMu = [t for (t,s) in dis.iteritems() if dis[t]== min(dis.values())][0]

            Clusters[MinMu].append( list(data[i,:]) )
    
        #Reformating Clusters  into N-dimensional array
        #Obtaining new centers
        for mu in range(K):
            Clusters[mu] = np.array(Clusters[mu])
            NewMu[mu] = np.mean(Clusters[mu], axis=0)

        #Discarding run and starting all over if a cluster has zero observations 
        if sum(sum(np.isnan(NewMu))) > 0:
            NewMu = np.zeros((K,2))
            OldMu = np.random.uniform(-1,1,(K,2))
        else:
            #Checking convergence of centers using an entrywise norm for matrices
            DistMu =   np.sqrt(sum([(norm(OldMu[mu] - NewMu[mu]))**2 for mu in range(K)]))
            OldMu = NewMu.copy()


    centers = {str(i) + 'mu': NewMu[i] for i in range( len(NewMu)) }
    return centers






    






#################################################################################################################################################################
#Experiment

#Initating experiment parameters and random seed
exp_length =  700
N=100 
d=2  
Total = 1000+1000
SVM_Ein = np.ones(exp_length)
SVM_Eout = np.zeros(exp_length)
RBF_Ein = np.zeros(exp_length)
RBF_Eout = np.zeros(exp_length)

#Coefficient in RBF and Number of Centers K plus Hard Margin for SVM and kernal = rbf
rho = 1.5
K = 12
C = 10**(5)
kernel = 'rbf'
gamma = rho
coef0=1
count = 0                             #Keeps track of the number of times data is not lineraly seperable 




#####Running Experiment 
for run in range(exp_length):

    while SVM_Ein[run]!=0:
        #Genterating the target function
        #Evaluating target funtion on training data
        X = np.random.uniform(-1,1,[Total,d])
        Y = np.sign(X[:,1] - X[:,0] + .25*np.sign(np.pi*X[:,0])) 
        y = Y[:N]
        x = X[:N,:]
    
        SVM_Solution = sk.svm.SVC(C = C,kernel = kernel, gamma = gamma, coef0=coef0, verbose=False).fit(x,y)
        #Estimating In-Sample error
        yEstimates = SVM_Solution.predict(x) 
        In_Accuracy = sk.metrics.zero_one_score(y,yEstimates)
        SVM_Ein[run] = 1 - In_Accuracy


        #Estimating Out-Sampple error
        YEstimates = SVM_Solution.predict(X[N:,:])
        Out_Accuracy = sk.metrics.zero_one_score(Y[N:],YEstimates)
        1 - Out_Accuracy
        SVM_Eout[run] = 1 - Out_Accuracy



###########################################################

#Performin RBF as a kernal for support vector machines

    #Performing regular radial basis functions 
    #Obtaining cluster means from K-means clustering
    centers = Centers(x,K)

    # Obtaining Q matrix consisting of the elements exp(-gamma*||X - mu||^2)
    #and a ones column 
    Q = Qmatrix(centers,K,x,[0,N],rho)


    #Pseudo-inverse to obtain rbf weights
    Pseudo_inverse = inv(dot(Q.transpose(),Q))
    w = dot(dot(Pseudo_inverse,Q.transpose()),y)
    #Obtaining Eout and Ein for Regular RBF Functions
    #In-Sample Performance for RBF
    g = np.sign(np.dot(Q,w))
    RBF_Ein[run] = np.sum(g!=y,dtype='float')/(N)

    #Out-Of-Sample Performance
    #Obtaining Q matrix for Test Dataset
    Q = Qmatrix(centers,K,X,[N,Total],rho)

    #Eout for RBF
    g = np.sign(np.dot(Q,w))
    RBF = np.sum(g!=Y[N:],dtype='float')/(Total-N)
    RBF_Eout[run] = RBF
  




print float(sum(SVM_Eout < RBF_Eout))/exp_length













    
        
    
    


    
    





   
    

    
    
    
    

 







    
        














