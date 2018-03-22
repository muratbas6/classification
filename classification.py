import numpy as np
import pandas as pd


my_data = pd.read_csv('data.txt',names=["x0","x1","y"]) #read the data


x = my_data.iloc[:,0:2]
ones = np.ones([x.shape[0],1])
x = np.concatenate((ones,x),axis=1)
print(x)
y = my_data.iloc[:,2:3].values
Q = np.random.rand(1,3)

alpha = 0.01
iters = 100000



def grad(x,y,iters,alpha,Q):
    for i in range(iters):
        Q = Q - (alpha / len(x)) * np.sum(x *((1 / (1 + np.exp(-(x @ Q.T)))) - y), axis=0)

    return Q

a = grad(x,y,iters,alpha,Q)
print(a)

predict = np.array([1,5,5])
print(1 / (1 + np.exp(-(predict @ a.T)))) # logistic hypothesis