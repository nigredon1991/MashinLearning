
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

data =  load_boston()
data1 =  scale(data.data)
kf = KFold(n = 506,shuffle = True, random_state=42, n_folds =5)
a = []




for p in np.linspace(1,10,200):
    print "p: " + p
    neigh = KNeighborsRegressor(n_neighbors=5 , weights='distance',p = p )
    a.append(np.mean(cross_val_score(neigh,data1,data.target, scoring='mean_squared_error', cv = kf)))
    print a[-1]

print  max(a)

