import pandas
dtrain = pandas.read_csv('perceptron-train.csv', header = None)
dtest = pandas.read_csv('perceptron-test.csv', header = None)

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X = dtrain[[1,2]]
y = dtrain[0]
clf = Perceptron(random_state= 241)
clf.fit(X, y)
predictions = clf.predict(X)

acc_not_scal =  accuracy_score(dtest[0], clf.predict(dtest[[1,2]]))
print acc_not_scal

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(dtrain[[1,2]])
X_test_scaled = scaler.transform(dtest[[1,2]])
clf1 = Perceptron(random_state= 241)
clf1.fit(X_train_scaled, dtrain[0])
acc_with_scal = accuracy_score(dtest[0],clf1.predict(X_test_scaled))
print acc_with_scal

print acc_with_scal - acc_not_scal