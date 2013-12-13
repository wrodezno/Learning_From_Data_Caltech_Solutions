import numpy as np






#Initating experiment parameters and random seed
#np.random.seed(0);
exp_length = 1000 ; N=100 ; d=2 ; 
iterations = np.zeros(exp_length)
Total = 100000
E_out = np.zeros(exp_length)



for i in range(exp_length):

    #Genterating the target function
    p1 = np.random.uniform(-1,1,d)
    p2 = np.random.uniform(-1,1,d)
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1]-m*p2[0]

    #Evaluating target funtion on training data
    X = np.random.uniform(-1,1,[Total,d])
    Logicals =  X[:,1] < X[:,0]*m+b
    Y = np.zeros(Total)
    Y[Logicals] = -1
    Y[Logicals==False] = 1
    y = Y[:N]

    #Adding a ones column to training data for perceptron algorithem
    constant = np.ones(Total)
    X = np.column_stack((constant,X))
    x = X[:N,:]


            
    #Initial weights of perceptron and hypothesis
    w = np.array([0,0,0])
    iteration = 1
    h = np.sign(np.dot(x,w))
    #Performing Perceptron algorithm
    while np.all(y==h)==False:
        j = np.random.randint(0,len(h))
        if h[j]!=y[j]:
            w = w + y[j]*x[j,:]
            iteration = iteration + 1
            h = np.sign(np.dot(x,w))

    iterations[i] = iteration

    #Accuracy of out of sample 
    
    g = np.sign(np.dot(X[N:,:],w))
    E_out[i] = np.sum(g!=Y[N:],dtype='float')/(Total-N)

Avg_E_out = np.average(E_out)   


    
Avg_Iterations = np.average(iterations)

print(Avg_Iterations )
print(Avg_E_out)



    
    

            
        





