

#Problem 4

Sim.Length = 1000
X <- vector()
g <- vector()
f <- vector()

for(i in 1:Sim.Length){
x <- runif(2,-1,1)
y <- sin(pi*x)

g <- c(g,as.vector(solve(t(x)%*%x)%*%(t(x)%*%y)))


X <- c(X,x)
f <- c(f,y)
}

g_bar <- mean(g)


#Problem 5
bias <-  mean((g_bar*X - f)^2)
#Problem 6
Expected.Value.D <- vector()
for(i in 1:length(g)){
Expected.Value.D <- c(Expected.Value.D,mean( (g[i]*X - g_bar*X)^2   ))
}
Expected.Value.X <- mean(Expected.Value.D)
variance <- Expected.Value.X
Eout <- bias + variance




#Problem 7
#Option A
Sim.Length = 1000
X <- vector()
g <- vector()
f <- vector()

for(i in 1:Sim.Length){
x <- runif(2,-1,1)
y <- sin(pi*x)
g <- c(g,(x[1]+x[2])/2)
X <- c(X,x)
f <- c(f,y)
}

g_bar <- mean(g)
bias <-  mean((g_bar*X - f)^2)
Expected.Value.D <- vector()
for(i in 1:length(g)){
Expected.Value.D <- c(Expected.Value.D,mean( (g[i]*X - g_bar*X)^2   ))
}
Expected.Value.X <- mean(Expected.Value.D)
variance = Expected.Value.X

Eout = bias + variance





#Option C
Sim.Length = 1000
X <- vector()
g <- vector()
f <- vector()
Xt <- vector()
slope <- vector()
constant <- vector()
for(i in 1:Sim.Length){
x <- runif(2,-1,1)
xt <- cbind(c(1,1),x)
y <- sin(pi*x)
reg <- as.vector(solve(t(xt)%*%xt)%*%(t(xt)%*%y))
constant <- c(constant,reg[1])
slope <- c(slope,reg[2])
X <- c(X,x)
Xt<-rbind(Xt,xt)
f <- c(f,y)
}




g_bar <- c(mean(constant),mean(slope))

bias <-  mean((Xt%*%g_bar - f)^2)



Expected.Value.D <- vector()
for(i in 1:length(slope)){
Expected.Value.D <- c(Expected.Value.D,mean( (Xt%*%c(constant[i],slope[i]) - Xt%*%g_bar)^2   ))
}
Expected.Value.X <- mean(Expected.Value.D)
variance = Expected.Value.X

Eout = bias + variance








#Option D
Sim.Length = 1000
X <- vector()
g <- vector()
f <- vector()
Xt <- vector()

for(i in 1:Sim.Length){
x <- runif(2,-1,1)
y <- sin(pi*x)
xt <- x^2

g <- c(g,as.vector(solve(t(xt)%*%xt)%*%(t(xt)%*%y)))


X <- c(X,x)
f <- c(f,y)
Xt <- c(Xt,xt)
}

g_bar <- mean(g)


#Problem 5
bias <-  mean((g_bar*Xt - f)^2)
#Problem 6
Expected.Value.D <- vector()
for(i in 1:length(g)){
Expected.Value.D <- c(Expected.Value.D,mean( (g[i]*Xt - g_bar*Xt)^2   ))
}
Expected.Value.X <- mean(Expected.Value.D)
variance <- Expected.Value.X
Eout <- bias + variance















#Option E
Sim.Length = 1000
X <- vector()
g <- vector()
f <- vector()
Xt <- vector()
slope <- vector()
constant <- vector()
for(i in 1:Sim.Length){
x <- runif(2,-1,1)
xt <- x^2
xt <- cbind(c(1,1),x)
y <- sin(pi*x)
reg <- as.vector(solve(t(xt)%*%xt)%*%(t(xt)%*%y))
constant <- c(constant,reg[1])
slope <- c(slope,reg[2])
X <- c(X,x)
Xt<-rbind(Xt,xt)
f <- c(f,y)
}




g_bar <- c(mean(constant),mean(slope))

bias <-  mean((Xt%*%g_bar - f)^2)



Expected.Value.D <- vector()
for(i in 1:length(slope)){
Expected.Value.D <- c(Expected.Value.D,mean( (Xt%*%c(constant[i],slope[i]) - Xt%*%g_bar)^2   ))
}
Expected.Value.X <- mean(Expected.Value.D)
variance = Expected.Value.X

Eout = bias + variance








