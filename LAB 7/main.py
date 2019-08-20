import numpy as num
import csv
from sklearn.svm import SVC
import matplotlib.pyplot as plt


with open("Grand-slams-men-2013.csv") as f:
    csv_list = list(csv.reader(f))

wnr1_list = num.array([])
wnr2_list = num.array([])
result_list = num.array([])
for row in csv_list:
    if row != csv_list[0]:
        wnr1_list = num.append(wnr1_list, int(row[12]))
        wnr2_list = num.append(wnr2_list, int(row[30]))
        result_list = num.append(result_list, int(row[3]))


X = num.vstack((wnr1_list, wnr2_list)).T
Y = result_list.T

X_Train = X[0:200, :]
X_Test = X[200:, :]
Y_Train = Y[0:200]
Y_Test = Y[200:]

Y_pred_1 = num.array([])
Y_pred_2 = num.array([])
Y_pred_3 = num.array([])

reg1 = SVC(kernel = 'linear')
reg2 = SVC(kernel = 'poly')
reg3 = SVC(kernel = 'rbf')

reg1.fit(X_Train,Y_Train)
reg2.fit(X_Train, Y_Train)
reg3.fit(X_Train, Y_Train)


Y_pred_1 = reg1.predict(X_Test)
Y_pred_2 = reg2.predict(X_Test)
Y_pred_3 = reg3.predict(X_Test)


X0 = num.array([])
X1 = num.array([])
X0A = num.array([])
X0B = num.array([])
X1A = num.array([])
X1B = num.array([])
ind = num.array([], dtype="int")
ind2 = num.array([], dtype="int")
ind3 = num.array([], dtype="int")

##numpy.delete(arr, obj, axis=None)[source]
##Return a new array with sub-arrays along an axis deleted. For a one dimensional array, this returns those entries not returned by arr[obj].

for i in range(len(Y_pred_1)) :
    if Y_pred_1[i] == 0:
        ind = num.append(ind, i)

X0 = X_Test[ind]
X1 = num.delete(X_Test, ind, 0)

for i in range(len(Y_pred_2)) :
    if Y_pred_2[i] == 0 :
        ind2 = num.append(ind2, i)

X0A = X_Test[ind2]
X1A = num.delete(X_Test, ind2, 0)

for i in range(len(Y_pred_3)):
    if Y_pred_3[i] == 0:
        ind3 = num.append(ind3, i)

X0B = X_Test[ind3]
X1B = num.delete(X_Test, ind3, 0)


plt. figure()
plt.scatter(X0[:, 0],X0[:, 1], color ="pink")
plt.scatter(X1[:, 0],X1[:, 1], color ="brown")

plt.figure()
plt.scatter(X0A[:, 0],X0A[:, 1], color= "yellow")
plt.scatter(X1A[:, 0],X1A[:, 1], color ="black")

plt.figure()
plt.scatter(X0B[:, 0],X0B[:, 1], color="green")
plt.scatter(X1B[:, 0],X1B[:, 1], color="orange")
plt.show()


