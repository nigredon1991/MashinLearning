# Neiron version 0.000000000000001
import numpy as np
import pandas


def f(x):
    return x

def nonlin(x,deriv=False):
    if(deriv==True):
        return f(x)*(1-f(x))
    return 1/(1+np.exp(-x))
# 8*98 = 784




class N1:
    "neiron with N inputs"

    def sigmoid(x,deriv=False):
        #standart activation function
        if(deriv==True):
            return (x*(1-x))
        return 1/(1+np.exp(-x))

    def __init__(self, numberW = 1,func = sigmoid):
        self.Weights = np.random.random(numberW).T
        self.func = func
        self.train_value = 0
    def predict(self,X):
        #direct direction
        self.train_value = self.func(np.dot(X,self.Weights))
        return self.train_value

    def learning(self,X, real_value, coef_learning = 0.1):
        #reverse direction
        delta = real_value - self.train_value
        self.Weights = self.Weights + coef_learning * delta * self.func(self.train_value,True) * X
        return delta

# 42000 * 784
import time
data = pandas.read_csv('NN/train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()

from sklearn.cross_validation import KFold
kf = KFold(n = 41999,shuffle = True, random_state=42, n_folds =2)
train_number = []
test_number  = []
for train,test in kf:
    train_number = train
    test_number  = test



moment = time.clock()
NN1 = [N1(98),N1(98),N1(98),N1(98),N1(98),N1(98),N1(98),N1(98)]
NN2 = [N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8)]
X2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#for i in train_number:
for i in (1,2):
    for k in range(1,1000):
        for j in range(1,8):
            #X2[j-1]= NN1[j].predict(X_train[X_train.columns[((j-1)*98):(j*98)]][i:(i+1)])
            X2[j-1] = NN1[j].predict(X_train[i][((j-1)*98):(j*98)])
        deltaN2 = np.array([1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,1.0])
        for j in range(0,9):
            NN2[j].predict(X2)
            if(y_train[i] == j):
                deltaN2[j] = NN2[j].learning(X2,1)
            else:
                deltaN2[j] = NN2[j].learning(X2,0)
        deltaN1 = np.array([0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0]) #8
        for j in range(0,7):
            for index in range(0,9):
                deltaN1[j] += deltaN2[index] * NN2[index].Weights[j]
        for j in range(1,8):
                #X2[j-1] = NN1[j].learning(X_train[X_train.columns[((j-1)*98):(j*98)]][i:(i+1)],deltaN2[j])
                X2[j-1] = NN1[j].learning(X_train[i][((j-1)*98):(j*98)],deltaN1[j])
        if max(deltaN2) < 0.1:
            k = 1001
print "time for learning: " + str(time.clock() - moment)

validation = 0
shape_test = 0
for i in test_number.shape:
    validation = i
shape_test = validation
for i in test_number:
    for j in range(1,8):
        #X2[j-1]= NN1[j].predict(X_train[X_train.columns[((j-1)*98):(j*98)]][i:(i+1)])
        X2[j-1] = NN1[j].predict(X_train[i][((j-1)*98):(j*98)])
    deltaN2 = np.array([1,1,1,1,1, 1,1,1,1,1])
    for j in range(0,9):
        NN2[j].predict(X2)
        if(y_train[i] != j):
            validation -= 1

print "Validation" + str (validation/shape_test)
