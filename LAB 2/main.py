import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt

with open("team_1.csv",encoding="Latin-1") as f:
    csv_list1 = list(csv.reader(f))

with open("team_2.csv", encoding="Latin-1") as f:
    csv_list2 = list(csv.reader(f))


def slope_intercept(x_val, y_val):
    xs = np.array(x_val)
    ys = np.array(y_val)
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) /
         ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    m = round(m, 2)
    b = (np.mean(ys) - np.mean(xs) * m)
    b = round(b, 2)

    return m, b

x = []
y = []

x1 = []
y1 = []

for player in csv_list1:
    if player != csv_list1[0]:
        age = int(player[4])
        exp = int(player[6])
        x.append(age)
        y.append(exp)

for player in csv_list2:
    if player != csv_list2[0]:
        age = int(player[4])
        exp = int(player[6])
        x1.append(age)
        y1.append(exp)

slope_intercept(x, y)
slope_intercept(x1, y1)

m, b = slope_intercept(x, y)
m1, b1 = slope_intercept(x1, y1)


reg_line = [(m*xs)+b for xs in x1]
reg_line_2 = [(m1*xs)+b1 for xs in x]

plt.figure(1)
plt.scatter(x1, y1, color="blue")
plt.plot(x1, reg_line, color="black")
plt.xlabel("Age")
plt.ylabel("Experience")

plt.figure(2)
plt.scatter(x, y, color="red")
plt.plot(x, reg_line_2, color="darkblue")
plt.xlabel("Age")
plt.ylabel("Experience")
plt.show()

RSS = 0
RSS_2 = 0

for i in range(len(y)):
    RSS = RSS + (y[i] - ((m1*x[i])+b1))**2
print("RSS 1: ", RSS)

for i in range(len(y1)):
    RSS_2 = RSS_2 + (y1[i] - ((m*x1[i])+b))**2

print("\nRSS 2: ", RSS_2)
