import pandas
import numpy as np

prices = pandas.read_csv('close_prices.csv')
djia = pandas.read_csv('djia_index.csv')

from sklearn.decomposition import PCA


clf = PCA(n_components=10)

clf.fit(prices[['AXP','BA','CAT','CSCO','CVX','DD','DIS','GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','T','TRV','UNH','UTX','V','VZ','WMT','XOM']])

sum = 0
for i in range(0,9):
    if (sum < 0.9):
        sum+=clf.explained_variance_ratio_[i]
        continue
    print i
    break
print sum

prices_new = clf.transform(prices[['AXP','BA','CAT','CSCO','CVX','DD','DIS','GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','T','TRV','UNH','UTX','V','VZ','WMT','XOM']])

print np.corrcoef(prices_new[:,0],djia['^DJI'])