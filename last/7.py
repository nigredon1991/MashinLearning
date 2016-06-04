import pandas
import numpy as np
import time
import datetime

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from random import randint

def Max_Min(a):
    max = 0
    maxIndex = 0
    min = 999999
    minIndex = 0 
    for i in range(len(a)):
        if a[i] > max:
            max = a[i]
            maxIndex = i
        if a[i] < min:
            min = a[i]
            minIndex = i
    return max, maxIndex,min, minIndex
 

features = pandas.read_csv('features.csv',index_col = 'match_id')
#find nan
for i in features.columns:
    for j in range(0,5):
        if(pandas.isnull(features[i][j])):
            print i
            break
#


y_train = features['radiant_win']
X_train = features.drop(features.columns[[-6,-5,-4,-3,-2,-1]],axis = 1)
X_train =  X_train.fillna(0)
true = True
while(true):
    kf = KFold(n = X_train.columns.shape[0],shuffle = True, n_folds =5)
    for p in range(10,30):
        #print "n_estimators: " ,p
        start_time = datetime.datetime.now()
        clf = GradientBoostingClassifier(n_estimators=p, learning_rate = 2)
        a = np.mean(cross_val_score(clf,X_train,y_train, scoring='roc_auc', cv = kf))
        #print 'Time elapsed:', datetime.datetime.now() - start_time
        if a > 0.7:
            print a
            for train, test in kf:
                print("%s %s" % (train, test))
            true = False
            
features = pandas.read_csv('features.csv',index_col = 'match_id')

y_train = features['radiant_win']
X_train = features.drop(features.columns[[-6,-5,-4,-3,-2,-1]],axis = 1)
X_train =  X_train.fillna(0)
kf = KFold(n = X_train.columns.shape[0],shuffle = True, n_folds =5)
for p in [10,20,30]:
	print "n_estimators: " + str(p)
	start_time = datetime.datetime.now()
	clf = GradientBoostingClassifier(n_estimators=p, learning_rate = 2)
	a = np.mean(cross_val_score(clf,X_train,y_train, scoring='roc_auc', cv = kf))
	print 'Time elapsed:', datetime.datetime.now() - start_time
	print a
        
################################################
#Logicstic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold

features = pandas.read_csv('features.csv',index_col = 'match_id')

y_train = features['radiant_win']
X_train = features.drop(features.columns[[-6,-5,-4,-3,-2,-1]],axis = 1)
X_train = X_train.fillna(0)
X_train = scale(X_train)


kf = KFold(len(X_train[0]),shuffle = True, random_state=42, n_folds =5)
a = []
first = 0.1 
end = 10
count_step = 200
for p in np.linspace(first,end,count_step):
    
    neigh = LogisticRegression(C = p,  n_jobs = 2)
    a.append(np.mean(cross_val_score(neigh,X_train,y_train, scoring='roc_auc', cv = kf)))
max, maxIndex,min, minIndex = Max_Min(a)
print 'C_max= ',first+(end-first)*maxIndex/count_step
print 'auc_score_max=', max 
print 'C_min= ',first+(end-first)*minIndex/count_step
print 'auc_score_min=', min 

##################### 
#Logicstic Regression without Heroes
features = pandas.read_csv('features.csv',index_col = 'match_id')

y_train = features['radiant_win']
X_train = features.drop(features.columns[[-6,-5,-4,-3,-2,-1]],axis = 1)
X_train = X_train.fillna(0)
X_train = X_train.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero', 'd1_hero', 'd2_hero' ,'d3_hero', 'd4_hero','d5_hero'],axis = 1)
X_train = scale(X_train)


kf = KFold(len(X_train[0]),shuffle = True, random_state=42, n_folds =5)
a = []
first = 0.1 
end = 10
count_step = 200
for p in np.linspace(first,end,count_step):
    
    neigh = LogisticRegression(C = p,  n_jobs = 2)
    a.append(np.mean(cross_val_score(neigh,X_train,y_train, scoring='roc_auc', cv = kf)))
max = 0
maxIndex = 0
for i in range(len(a)):
    if a[i] > max:
        max = a[i]
        maxIndex = i
print 'C = ',first+(end-first)*maxIndex/count_step
print 'auc_score=', max 

##########################
#Logicstic Regression with mod Heroes

features = pandas.read_csv('features.csv',index_col = 'match_id')

y_train = features['radiant_win']
X_train = features.drop(features.columns[[-6,-5,-4,-3,-2,-1]],axis = 1)
X_train = X_train.fillna(0)
X_train = X_train.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero', 'd1_hero', 'd2_hero' ,'d3_hero', 'd4_hero','d5_hero'],axis = 1)

N = 112
X_pick = np.zeros((features.shape[0], N))
for i, match_id in enumerate(features.index):
    for p in xrange(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_train = np.concatenate((X_train, X_pick), axis=1)

X_train = scale(X_train)

kf = KFold(len(X_train[0]),shuffle = True, random_state=42, n_folds =5)
a = []
first = 0.1 
end = 10
count_step = 200
for p in np.linspace(first,end,count_step):
    
    neigh = LogisticRegression(C = p,  n_jobs = 2)
    a.append(np.mean(cross_val_score(neigh,X_train,y_train, scoring='roc_auc', cv = kf)))
max = 0
maxIndex = 0
for i in range(len(a)):
    if a[i] > max:
        max = a[i]
        maxIndex = i
print 'C = ',first+(end-first)*maxIndex/count_step
print 'auc_score=', max 

