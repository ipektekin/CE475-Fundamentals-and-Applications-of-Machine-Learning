import csv
import matplotlib.pyplot as plt
import numpy as num
import pandas

x1 = []
x2 = []
x3 = []
y = []


with open("teams_comb.csv",encoding="Latin-1") as f:
    csv_list = list(csv.reader(f))

for player in csv_list:
    if player != csv_list[0]:
        age = int(player[4])
        exp = int(player[6])
        pow = float(player[7])
        sal = int(player[8])

        x1.append(age)
        x2.append(exp)
        x3.append(pow)
        y.append(sal)



x0 = num.ones((1, len(x1)))

for a in range(len(x1)):

    X = num.vstack((x0, x1, x2, x3)).T



def coef(X,y):
    a1 = X.T
    a2 = num.dot(a1, a1.T)
    a3 = num.linalg.inv(a2)
    a4 = num.dot(a3, a1)
    B = num.dot(a4, y)
    return B


def predict(X,B):
    Y = num.dot(X, B)
    return Y

#print(predict(X,coef(X,y)))
#print(X)
locvpredict=[]
test=[]

for a in range(len(X)):
    test_x = X[a]
    test_y = y[a]
    train_x= num.delete(X,a,0)
    train_y = num.delete(y,a,0)
    Z=predict(test_x,coef(train_x,train_y))
    locvpredict.append(Z)
    test.append(test_y)


ar_predict=num.array(locvpredict)
ar_test=num.array(test)

MES = ar_predict -ar_test
MES_sqr = MES**2
lovc_mse = num.mean(MES_sqr)

#TASK 2

task2predict=predict(X,coef(X,y))
task2error = task2predict-y
task2mse = num.mean(task2error**2)


print(lovc_mse)
print(task2mse)

plt.scatter(ar_predict,MES)
plt.scatter(task2predict,task2error)
plt.hlines(y=0, xmin = 0, xmax = 50000)
plt.show()



""""

Y = num.dot(X, B)

U = Y - y
print(U)

plt.scatter(Y, U)
plt.hlines(y=0, xmin = 0, xmax = 50000)
plt.show()
"""