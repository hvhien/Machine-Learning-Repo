import numpy as np
import matplotlib.pyplot as plt

raw = np.loadtxt('univariate.txt',delimiter = ',')

X = np.copy(raw)
X[:,1] = X[:,0]
X[:,0] = 1
y = raw[:,1]

def predict(X,theta):
    return X@theta

def computeCost(X,y,Theta):
    error = predict(X,Theta)
    m = np.size(y)
    J = (1/(2*m)) * np.transpose(error)@error
    return J

def gradientdescent(X,y,alpha=0.02,iter=5000):
    theta = np.zeros(np.size(X,1))
    m = np.size(y)
    X_T = np.transpose(X)
    precost = computeCost(X,y,theta)
    for i in range(0,iter):
        error = predict(X,theta) - y
        theta = theta - (alpha/m) * (X_T@error)
        cost = computeCost(X,y,theta)
        if np.round(cost,15) == np.round(precost,15):
            break
        precost = cost
    yield theta

[Theta] = gradientdescent(X,y)

predicted = X @ Theta
plt.plot(X[:,1:],y,'rx')
plt.plot(X[:,1:],predicted,'b')
plt.show()
