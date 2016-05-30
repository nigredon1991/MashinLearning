from   sklearn.cluster import KMeans
import pandas
from skimage.io import imread
image = imread('parrots.jpg')
import math
def MSE(I,K,m,n):
    a = 0.0
    for i in range(0,m-1):
        for j in range(0,n-1):
            a+= math.fabs(I[i][j][0]- K[i][j][0])*math.fabs(I[i][j][0]- K[i][j][0])
            a+= math.fabs(I[i][j][1]- K[i][j][1])*math.fabs(I[i][j][1]- K[i][j][1])
            a+= math.fabs(I[i][j][2]- K[i][j][2])*math.fabs(I[i][j][2]- K[i][j][2])
    return a/(m*n)
def PSNR(I,K,m,n):
    return 20*math.log10(I.max()/MSE(I,K,m,n))

from skimage import img_as_float
iaf = img_as_float(image,force_copy=True)
m = iaf.shape[0]
n = iaf.shape[1]
def GetColor(x,m,n,k):
    y = np.zeros((m,n), dtype= np.float64)
    for i in range(0,m-1):
        for j in range(0,n-1):
            y[i][j]=x[i][j][k]
    return y
iafR = GetColor(iaf,m,n,0)
iafG = GetColor(iaf,m,n,1)
iafB = GetColor(iaf,m,n,2)

Rclf = KMeans(init='k-means++',random_state=241)
Rpred = Rclf.fit_predict(iafR)

Gclf = KMeans(init='k-means++',random_state=241)
Gpred = Gclf.fit_predict(iafG)


Bclf = KMeans(init='k-means++',random_state=241)
Bpred = Gclf.fit_predict(iafB)


import matplotlib.pyplot as plt
plt.imsave(iaf,'1')

###############################

from   sklearn.cluster import KMeans
import pandas
import numpy as np
from skimage.io import imread
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 8
image = imread('parrots.jpg')
#image = np.array(image, dtype=np.float64) / 255
from skimage import img_as_float
image = img_as_float(image)
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))


print("Fitting model on a small sub-sample of the data")
t0 = time()
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
print("done in %0.3fs." % (time() - t0))

print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image ')
#plt.imshow(image)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

for f in range(2,20):
    n_colors = f
    t0 = time()
    kmeans = KMeans(n_clusters=n_colors, random_state=241).fit(image_array)
    print("done in %0.3fs." % (time() - t0))

    #print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    new_image = recreate_image(kmeans.cluster_centers_, labels, w, h)
    print("done in %0.3fs." % (time() - t0))
    print "PSNR for i = " + str(f) + ": " + str(PSNR(image,new_image,w,h))


for f in range(2,20):
    n_colors = f
    t0 = time()
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    print("done in %0.3fs." % (time() - t0))

    #print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    new_image = recreate_image(find_average(image_array,labels,f), labels, w, h)
    print("done in %0.3fs." % (time() - t0))
    print "PSNR for i = " + str(f) + ": " + str(PSNR(image,new_image,w,h))


def find_average(array,labels,n):
    sum = np.zeros((n-1,3),dtype = np.float64)
    index = np.zeros((n-1,3),dtype = np.float64)
    for k in range(array.shape[0]):
      #  print k
      #  print labels[k*3]
      #  print array[k*3]
        sum[labels[k]][0] += array[k][0]
        index[labels[k]][0]+= 1

        sum[labels[k]][1] += array[k][1]
        index[labels[k]][1]+= 1

        sum[labels[k]][2] += array[k][2]
        index[labels[k]][2]+= 1
    print sum/index
