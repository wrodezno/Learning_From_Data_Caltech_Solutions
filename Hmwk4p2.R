

dvc = 50
mH2 = (2*N)^dvc
mH = (N)^dvc
delta = .05

N = seq(from = 100, to = 20000, by = 10)
#N=20


Original.VC.Boound = sqrt((8/N)*log(4*(mH2)/.05))

Rademacher.Penalty.Bound = sqrt(2*log(2*N*(mH))/N) + sqrt((2/N)*log(1/delta)) + (1/N)

#***************************************

Devroye <- c(rep(0,length(N)))
i <- 1 
for(n in N){
mH.Square <- (n^2)^dvc
mu = log(4*mH.Square/delta)
root = polyroot(c(-mu,-2/n,(n-2)/n))[1]
Devroye[i] <- root
i <- i + 1
}


#*********************



Parr.VandenBroek <- c(rep(0,length(N)))
j <- 1

for(n in N){
mH2 = (2*n)^dvc
q <- (1/n)*log(6*mH2/delta)
Parr.VandenBroek [j] <- polyroot(c(-q,-2/n,1))[1]
j <- j+1
}




plot(N,as.numeric(Original.VC.Boound), col = 'red', typ = 'l',ylim = c(0,9))

lines(N,as.numeric(Rademacher.Penalty.Bound),col = 'blue')

lines(N,as.numeric(Devroye), col = 'yellow')

lines(N,as.numeric(Parr.VandenBroek), col = 'purple')





