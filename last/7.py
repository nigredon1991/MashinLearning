import pandas
import numpy as np
import time
import datetime

print 'Time elapsed:', datetime.datetime.now() - start_time
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from random import randint

def NotEq(x,k):
	for i in range(0,k):
		for j in range(i+1,k):
			#print(j)
			if x[i] == x[j]:
				print x[i]
				x[j]+= 1

features = pandas.read_csv('features.csv',index_col = 'match_id')

for i in range(10):
	k = 10
	r = []
	for j in range(k):
		r.append(randint(0,101))
	NotEq(r,k)
	print r
	features1 = features.drop(features.columns[r],axis = 1)
	y_train = features['radiant_win']
	X_train = features.drop(features.columns[[-6,-5,-4,-3,-2,-1]],axis = 1)
	X_train =  X_train.fillna(0)
	kf = KFold(n = X_train.columns.shape[0],shuffle = True, n_folds =5)
	for p in [10,20,30]:
		#print "n_estimators: " + str(p)
		start_time = datetime.datetime.now()
		clf = GradientBoostingClassifier(n_estimators=p, learning_rate = 2)
		a = np.mean(cross_val_score(clf,X_train,y_train, scoring='roc_auc', cv = kf))
		#print 'Time elapsed:', datetime.datetime.now() - start_time
		print a

################################################

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold

features = scale(features)
features_test = scale(features_test)

kf = KFold(n = 506,shuffle = True, random_state=42, n_folds =5)
a = []
for p in np.linspace(1,10,200):
    print "p: " + p
    neigh = LogisticRegression()
    a.append(np.mean(cross_val_score(neigh,data1,data.target, scoring='mean_squared_error', cv = kf)))
    print a[-1]

print  max(a)
