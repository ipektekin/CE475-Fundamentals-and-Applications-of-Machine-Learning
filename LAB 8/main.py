from PIL import Image
import numpy as num

image = Image.open("black white flower.jpg")
image = image.convert(mode='1')
image = image.resize((320, 256))

x_coor = num.array([])
y_coor = num.array([])

for i in range(320):
    for j in range(256):
        if image.getpixel((i, j)) == 255:
            x_coor = num.append(x_coor, i)
            y_coor = num.append(y_coor, j)

X = num.vstack((x_coor, y_coor)).T

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
predictX = kmeans.predict(X)

firstCluster_x = num.array([])
secondCluster_x = num.array([])
thirdCluster_x = num.array([])
fourthCluster_x = num.array([])

firstCluster_y = num.array([])
secondCluster_y = num.array([])
thirdCluster_y = num.array([])
fourthCluster_y = num.array([])



for i in range(len(predictX)):

    if kmeans.labels_[i] == 0:
        firstCluster_x = num.append(firstCluster_x, x_coor[i])
        firstCluster_y = num.append(firstCluster_y, y_coor[i])

    if kmeans.labels_[i] == 1:
        secondCluster_x = num.append(secondCluster_x, x_coor[i])
        secondCluster_y = num.append(secondCluster_y, y_coor[i])

    if kmeans.labels_[i] == 2:
        thirdCluster_x = num.append(thirdCluster_x, x_coor[i])
        thirdCluster_y = num.append(thirdCluster_y, y_coor[i])

    if kmeans.labels_[i] == 3:
        fourthCluster_x = num.append(fourthCluster_x, x_coor[i])
        fourthCluster_y = num.append(fourthCluster_y, y_coor[i])


import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x_coor,y_coor , color ='black')
plt.figure()
plt.scatter(firstCluster_x, firstCluster_y)
plt.scatter(secondCluster_x, secondCluster_y)
plt.scatter(thirdCluster_x,thirdCluster_y)
plt.scatter(fourthCluster_x,fourthCluster_y)
plt.show()
