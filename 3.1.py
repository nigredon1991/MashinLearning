from sklearn.svm import SVC
import pandas
data = pandas.read_csv('svm-data.csv', header = None)
clf = SVC( kernel='linear', C = 100000, random_state = 241)
X = data[0]
y = data[[1,2]]
clf.fit(y,X)
print clf.support_


from sklearn.cross_validation import KFold
from sklearn import datasets
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
import numpy as np
from sklearn.svm import SVC
import pandas as pd

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

clfV = TfidfVectorizer(analyzer= 'word',stop_words = 'english', min_df=1 , max_df= 10 )
X =  clfV.fit_transform(newsgroups.data)
#y = clfV.transform(newsgroups.target)
#X = newsgroups.data
y = newsgroups.target


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
#clf.fit(X, y)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv, n_jobs=-1)
gs.fit(X, y)

for a in gs.grid_scores_:
    print a.mean_validation_score# — оценка качества по кросс-валидации
    print a.parameters# — значения параметров

clf = SVC( kernel='linear', C = 1 , random_state = 241)
clf.fit(X,y)
a = []
for t in pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index :
    a.append(clfV.get_feature_names()[t])
print ', '.join(sorted(a))


clf = SVC( kernel='linear', C = 0 , random_state = 241)
clf.fit(X,y)
a = []
for t in pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index :
    a.append(clfV.get_feature_names()[t])
print ' '.join(sorted(a))



import heapq
#print pd.Series(clfN.coef_.toarray().reshape(-1)).abs().nlargest(10).index

a = []
for t in pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index :
    a.append(clfV.get_feature_names()[t])
print ' '.join(sorted(a))

for t in range(0,len(clfV.get_feature_names())):
    if (clfV.get_feature_names()[t][0] == 'b'):
        print clfV.get_feature_names(t)

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
    print("%s: %s" % (category, " ".join(feature_names[top10])))
