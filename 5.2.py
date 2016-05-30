import pandas
import numpy as np

gbm = pandas.read_csv('gbm-data.csv')
Xgbm = gbm.drop(gbm.columns[0],axis = 1)
ygbm = gbm['Activity']

X = Xgbm.values
y = ygbm.values


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.8,random_state=42)

from sklearn.ensemble import GradientBoostingClassifier
import math
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


clfR = RandomForestClassifier(n_estimators=250,verbose=True, random_state=241)
clfR.fit(X_train,y_train)
print log_loss(y_test, clfR.predict_proba(X_test))

########################################


clf = GradientBoostingClassifier(n_estimators=250,verbose=True, random_state=241,learning_rate = 0.2)
clf.fit(X_train,y_train)

sdf = []
k = 0
for y_pred in  enumerate(clf.staged_decision_function(X_test)):
    sdf.append([])
    for i in y_pred[1]:
        sdf[k].append( 1 / (1 + math.exp(-1* i )))
        #sdf[k].append( i + 1-1)
    k+= 1

k = 0
for i  in sdf:
    print (str(k)+ " " + str( log_loss(y_true = y_test, y_pred =  i)))
    k+=1
a= []
for i  in sdf:
    a.append(log_loss(y_true = y_test, y_pred =  i))
print min(a)

#pred = 1 / (1 + math.exp(-1* y_pred ))
clf.predict_proba(X_test)



####################################################
learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
original_params = {'n_estimators': 250, 'verbose': True, 'random_state': 241}
plt.figure()
for label, color, setting in [('learning_rate= 1', 'orange',
                               {'learning_rate': 1.0}),
                              ('learning_rate=0.5', 'turquoise',
                               {'learning_rate': 0.5}),
                              ('subsample=0.3', 'blue',
                               {'learning_rate': 0.3}),
                              ('learning_rate=0.2', 'gray',
                               {'learning_rate': 0.2}),
                              ('learning_rate=0.1', 'magenta',
                               {'learning_rate': 0.1})]:
    params = dict(original_params)
    params.update(setting)

    clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = clf.loss_(y_test, y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)


plt.savefig(str(i)+'example.png')
plt.show()

