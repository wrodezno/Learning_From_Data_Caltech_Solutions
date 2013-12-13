#Wilfedo's Python Code for Learning from Data Hmwk_1 Problem_1
#10/8/2013

#Libraries used for program
import numpy as np

#np.random.seed(98)
#Setting initial variables 
Sim_Length = 100000
Num_Coins = 1000
Num_Flips = 10
v1 = 0
StoreDiff = np.zeros(Sim_Length)
Epsilon = 4.81
Flip_Results = np.zeros(Num_Coins)
p = .5   #Probablity of getting a heads


for i in range(Sim_Length):
        Flip_Results = np.random.binomial(Num_Flips, p, Num_Coins)     #Result of fliping a 1000 coins 10 times 
        V1 = float(Flip_Results[0])/Num_Flips

        if np.absolute(V1-p) > Epsilon:
            StoreDiff[i] = 1






bound = 2*(Epsilon**(-2*(Epsilon**(2))*Num_Flips))
print(np.sum(StoreDiff)/Sim_Length)



        
        
    
    
