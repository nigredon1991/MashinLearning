import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from sklearn.metrics import r2_score


abalon = pandas.read_csv('abalone.csv')
abalon['Sex'] = abalon['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = abalon[['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']]
y = abalon['Rings']
for i in range(1,50):
    clf = RandomForestRegressor(n_estimators=i,random_state=1)
    clf.fit(X, y)
    kf = KFold(n = 4176,shuffle = True, random_state=1, n_folds =5)
    print i
    print np.mean(cross_val_score(estimator=clf,X =  X,y = y, scoring = 'r2',cv=kf))
    #print r2_score(clf.predict(X),y)
