import pandas

train = pandas.read_csv('salary-train.csv')
train['FullDescription'] = train['FullDescription'].str.lower()
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)



from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import  TfidfVectorizer

train_test = pandas.read_csv('salary-test-mini.csv')

clFV = TfidfVectorizer(min_df = 5)
train1 =  clFV.fit_transform(train['FullDescription'])
train1_test = clFV.transform(train_test['FullDescription'])


train_test['FullDescription'] = train_test['FullDescription'].str.lower()
train_test['FullDescription'] = train_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


train_test['LocationNormalized'].fillna('nan', inplace=True)
train_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()

X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(train_test[['LocationNormalized', 'ContractTime']].to_dict('records'))


from scipy.sparse import hstack
X = hstack([X_train_categ,train1])
test = hstack([X_test_categ,train1_test])

from sklearn.linear_model import Ridge

clf = Ridge(alpha=1,random_state=241,normalize=True)
clf.fit(X,train['SalaryNormalized'])
print clf.predict(test)