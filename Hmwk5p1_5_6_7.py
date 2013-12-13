#Author: Wilfredo Rodezno, Jr.
#Purpose: Caltech Learning from Data Hmwk5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Problem 1
sigma = .1
d = 8
bound  = .008
N = np.array([10,25,100,500,1000],dtype = 'float')


Expected_Ein = (sigma**(2))*(1- ((d+1)/N))

for i in range(len(N)):
    if Expected_Ein[i] >= bound:
       print(N[i])
        break
        

#Problem 5

#initial variables 
var = np.array([1,1])
nu = .1
error  = (var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))**(2)
bound = 10**(-14)
iterations = 0 



#Batch Gradient Descent Algorithm

while error >= bound:
#for i in range(15):
    #Computing the Gradiant
    Du = 2*(np.exp(var[1]) + 2*var[1]*np.exp(-var[0]))*(var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))
    Dv = 2*(var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))*(var[0]*np.exp(var[1])-2*np.exp(-var[0]))

    Gradient = np.array([Du,Dv])

    #Gradient Descent upgrade 
    var = var - nu*Gradient
    error  = (var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))**(2)
    #Upgrading the number of iterations
    iterations+=1




#Problem 6

Options  = [np.array([1.0, 1.0]), np.array([0.713,0.045]), np.array([0.016,0.112]), np.array([-.083, .029]), np.array([.045,.024])]

Distances = [np.linalg.norm(var-x) for x in Options]

MinDistance, index = min([  (d,i) for (i,d) in enumerate(Distances) ])






#Problem 7

#initial variables 
var = np.array([1,1])
nu = .1
error  = (var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))**(2)
bound = 10**(-14)
iterations = 0 



#Coordinate Gradient Descent Algorithm


for i in range(15):

    #Move along U
    Du = 2*(np.exp(var[1]) + 2*var[1]*np.exp(-var[0]))*(var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))
    var = var - nu*np.array([Du,0])

    #Move along v
    Dv = 2*(var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))*(var[0]*np.exp(var[1])-2*np.exp(-var[0]))
    var = var - nu*np.array([0,Dv])

    #Updating Error
    error  = (var[0]*np.exp(var[1]) - 2*var[1]*np.exp(-var[0]))**(2)



















