import numpy as num
import matplotlib.pyplot as plt
import csv

x1 = []
x2 = []
x3 = []
y = []


with open("team.csv",encoding="Latin-1") as f:
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



x0 = num.ones(len(x1), dtype=int)

#Transpose
X = num.array([x0, x1, x2, x3]).T

a1 = X.T
a2 = num.dot(a1, a1.T)
a3 = num.linalg.inv(a2)
a4 = num.dot(a3, a1)
B = num.dot(a4, y)

Y = num.dot(X, B)

U = Y - y
print(U)

plt.scatter(Y, U)
plt.hlines(y=0, xmin = 0, xmax = 20000)
plt.show()