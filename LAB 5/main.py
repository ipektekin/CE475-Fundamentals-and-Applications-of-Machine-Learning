import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Grand-slams-men-2013.csv")

data_x=data["NPA.1"]
data_y=data["UFE.2"]

x_arr = data_x
y_arr = data_y

out_arr= x_arr.argsort()
x = np.array(x_arr[out_arr]).reshape(-1,1)
y = np.array(y_arr[out_arr]).reshape(-1,1)

knot1 =[10, 20, 30]
knot2 =[20,30]
knot3= [30]


def coef(X,y):
    a1 = X.T
    a2 = np.dot(a1, a1.T)
    a3 = np.linalg.inv(a2)
    a4 = np.dot(a3, a1)
    B = np.dot(a4, y)
    return B


def predict(X,B):
    Y = np.dot(X, B)
    return Y

def knot(x,y,knot):
    x0 = np.ones(len(x)).reshape(-1,1)
    x1 = np.append(x0, x, axis=1)
    x2 = np.append(x1, x**2, axis=1)
    x3 = np.append(x2, x**3, axis=1)
    z = np.zeros(len(x)).reshape(-1,1)
    z1 = np.zeros(len(x)).reshape(-1,1)
    for i in range(len(knot)-1):
        z = np.append(z,z1,axis=1)
    for i in range(len(x)):
        for j in range(len(knot)):
            if x[i]>knot[j]:
                z[i][j] = (x[i]-knot[j])**3
    x4 = np.append(x3,z,axis=1)
    B = coef(x4,y)
    Yhat = predict(x4,B)
    return Yhat

prediction1=knot(x,y,knot1)

prediction2=knot(x,y,knot2)

prediction3=knot(x,y,knot3)

plt.scatter(x,y)
plt.plot(x,prediction1)
plt.plot(x,prediction2)
plt.plot(x,prediction3)
plt.show()
