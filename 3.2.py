import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
import math
data = pandas.read_csv('data-logistic.csv', header = None)

def grad_w1(w1,w2,k , x1, x2, y , C ):
    a=0.0
    for i in range(0,x1.size):
        a+= x1[i]*y[i]*(1- 1/(1+np.exp(-y[i]*(w1*x1[i]+w2*x2[i]) ) ) )
    return (w1 + k*a/x1.size - k * C *w1 )


def grad_w2(w1,w2,k , x1, x2, y , C  ):
    a=0.0
    for i in range(0,x1.size):
        a+= x2[i]*y[i]*(1- 1/(1+np.exp(-y[i]*(w1*x1[i]+w2*x2[i]))) )
    return (w2 + k*a/x1.size - k * C *w2 )

def mymod(x):
    if(x<0):
        return -x
    return x

def calc_w(w1,w2,C,x1,x2,y,k):
    for i in range(1,10000):
        print w1,w2
        tw1 = grad_w1(w1,w2,k,x1,x2,y,C)
        tw2 = grad_w2(w1,w2,k,x1,x2,y,C)
        #print tunLogRegr(x1,x2,y,w1,w2)-tunLogRegr(x1,x2,y,tw1,tw2) < 1e-5
        #if(mymod(tunLogRegr(x1,x2,y,w1,w2)-tunLogRegr(x1,x2,y,tw1,tw2)) < 1e-5):
        if(mymod(w2 - tw2)+ mymod(w1 - tw1) < 1e-5):
            print i
            break
        w1 = tw1
        w2 = tw2
    return w1,w2

y = data[0]
x1 = data[1]
x2 = data[2]
C = 0.0
k = 0.1
w1,w2 = 1,1
w1,w2 = calc_w(w1,w2,C,x1,x2,y,k)
print w1,w2

C = 10.0
w1r,w2r =  1,1
w1r,w2r = calc_w(w1r,w2r,C,x1,x2,y,k)
print w1r, w2r

my=[]
for i in range(0, x1.size):
    my.append(1/(1+ np.exp(-x1[i]*w1 - x2[i]*w2)))

myr=[]
for i in range(0,x1.size):
    myr.append(1/(1+ np.exp(-x1[i]*w1r - x2[i]*w2r)))


print roc_auc_score(y,my)

print roc_auc_score(y,myr)



def tunLogRegr(x1,x2,y,w1,w2):
    sumexp = 0.0
    for i in range(0,x1.size):
        sumexp += np.log10(1 + np.exp(-y[i]*(w1*x1[i] + w2 * x2[i])))
    return (sumexp/x1.size)


def get_coef1(x1,x2,y,w1,w2):
    a=0.0
    for i in range(0,x1.size):
        a+= x1[i]*y[i]*(1- 1/(1+np.exp(-y[i]*(w1*x1[i]+w2*x2[i]) ) ) )
    return a

def get_coef2(x1,x2,y,w1,w2):
    a=0.0
    for i in range(0,x1.size):
        a+= x2[i]*y[i]*(1- 1/(1+np.exp(-y[i]*(w1*x1[i]+w2*x2[i]))) )
    return a
