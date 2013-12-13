#Wilfedo's Python Code for Learning from Data Hmwk_2 Problem_1
#10/8/2013

#Libraries used for program
import numpy as np

#np.random.seed(98)
#Setting initial variables 
Sim_Length = 100000
Num_Coins = 1000
Num_Flips = 10
V1 = 0
Vmin = 0
vrand = 0
Sum_V1 = 0
Sum_Vmin = 0
Sum_Vrand = 0
Flip_Results = np.zeros(Num_Coins)
p = .5                                     #Probablity of getting a heads


for i in range(Sim_Length):
        Flip_Results = np.random.binomial(Num_Flips, p, Num_Coins)     #Result of fliping a 1000 coins 10 times 
        V1 = float(Flip_Results[0])/Num_Flips
        Vmin = float(Flip_Results.min())/Num_Flips
        Vrand = float(Flip_Results[np.random.randint(Num_Coins)])/Num_Flips

        Sum_V1 = Sum_V1 + V1
        Sum_Vmin = Sum_Vmin + Vmin
        Sum_Vrand = Sum_Vrand + Vrand



avg_V1 = Sum_V1/Sim_Length
avg_Vmin = Sum_Vmin/Sim_Length
avg_Vrand = Sum_Vrand/Sim_Length

print(avg_Vmin)




        
        
    
    
