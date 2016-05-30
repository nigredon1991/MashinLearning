import pandas
from sklearn.svm import SVC
data = pandas.read_csv('wine.data')

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd


kf = KFold(n = 178,shuffle = True, random_state=42, n_folds =5)

for train_index, test_index in kf:
    print ("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]

data[14:15] # строки
data['s1'] # столбцы
data.iget_value(0,0) # элемент

neigh.fit(y = data['s1'] ,X =  data.groupby(['s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']).head())

clf = svm.SVC(kernel='linear', C=1)


import pandas as pd
data_train = pd.DataFrame()
data_test = pd.DataFrame()

for p in range(1,50):
    print p
    neigh = KNeighborsClassifier(n_neighbors =p)
    data_test = data
    data_train = data
    a = []
    for train_index, test_index in kf:
        for x in train_index:
            data_test = data_test.drop(x)
        for y in test_index:
            data_train = data_train.drop(y)
        neigh.fit(X = data_test.groupby(['s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']).head(), y = data_test['s1'])
        a.append(
        np.mean(cross_val_score(estimator = neigh, y = data_train['s1'] ,X =  data_train.groupby(['s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']).head(), scoring= 'accuracy', cv = kf))
        )
        data_test = data
        data_train = data
    print np.mean(a)



for p in range(1,50):
    print p
    neigh = KNeighborsClassifier(n_neighbors =p)
    print np.mean(cross_val_score(estimator = neigh, y = data['s1'] ,X =  data.groupby(['s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']).head(), scoring= 'accuracy', cv = kf))

data1 = pd.DataFrame()
for s in ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']:
        data1[s] = scale(data[s])

data1 =  scale(data)


for p in range(1,50):
    print p
    neigh = KNeighborsClassifier(n_neighbors =p)
    print np.mean(cross_val_score(estimator = neigh, y = data1['s1'] ,
                                  X =  data1.groupby(['s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']).head(),
                                  scoring= 'accuracy', cv = kf))


df = pd.DataFrame(data1)
for p in range(1,50):
    print p
    neigh = KNeighborsClassifier(n_neighbors =p)
    print np.mean(cross_val_score(
        estimator = scale(neigh), y = df[:,0] ,X =  df[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]],
        scoring= 'accuracy', cv = scale(kf)))



