import pandas
import numpy as np
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('data-logistic.csv', header = None)


k = 0.1
X = data[[1,2]]
y = data[0]
w = np.array([0,0])


def delta(X,y,w,c,k):
    dp = np.einsum('ij,j->i',X,w)
    coeff = 1-(1/(1+np.exp(-y*dp)))
    a = k/coeff.shape[0]*(np.einsum('ij,i,i->j',X,y,coeff))
    w = w + a - k*c*w
    return w

c = 0
wn = delta(X,y,w,c,k)
c = 10
wnr = delta(X,y,w,c,k)

wnr = [0.0285594315239, 0.024780878997]

wn = [0.288108116638, 0.0917091725253]


my = 1/(1+ np.exp(-X[1]*wn[0] - X[2]*wn[1]))
myr = 1/(1+ np.exp(-X[1]*wnr[0] - X[2]*wnr[1]))

print roc_auc_score(y,my)

print roc_auc_score(y,myr)
