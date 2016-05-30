import pandas
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
features = pandas.read_csv('features.csv',index_col = 'match_id')
features_test = pandas.read_csv('features_test.csv',index_col = 'match_id')
y_train = features['radiant_win']
# y_test = features_test['radiant_win']
X_train = features.drop(features.columns[[102,103,104,105,106,107]],axis = 1)
X_test = features_test

X_train =  X_train.fillna(0)

kf = KFold(n = X_train.columns.shape[0],shuffle = True, n_folds =5)
a = []
for p in [10,15,20,25,30]:
    print "n_estimators: " + str(p)
    clf = GradientBoostingClassifier(n_estimators=p, learning_rate = 2)
#    clf.fit(X_train,y_train)
    a.append(np.mean(cross_val_score(clf,X_train,y_train, scoring='roc_auc', cv = kf)))
    print a[-1]




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
