#
import numpy as np
import pandas


def f(x):
    return x

def nonlin(x,deriv=False):
    if(deriv==True):
        return f(x)*(1-f(x))
    return 1/(1+np.exp(-x))
    

def MyMod(x):
    if(x<0):
        return -x
    else:
        return x

def sigmoid(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

import json

class NN:
    "neiral network direct propagation"
    def __init__(self, size_in, size_hidden, size_out):
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.input = np.ones(size_in)
        self.weights0 = np.random.random([size_in,size_hidden])-0.5
        self.delta_hidden = np.zeros(size_hidden)
        self.free_mem_h = np.ones(size_hidden)*0.5
        self.train_values_h = np.zeros(size_hidden)####
        self.weights1 = np.random.random([size_hidden, size_out])-0.5
        self.delta_out = np.zeros(size_out)
        self.free_mem_out = np.ones(size_out)*0.5
        self.train_values_out = np.zeros(size_out)

    def predict(self,X):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        ######
        for i in range(0,self.size_in ):
            for j in range(0,self.size_hidden ):
                self.train_values_h[j] += self.weights0[i][j]*X[i]
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] += self.free_mem_h[i]
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        ##########
        for i in range(0,self.size_hidden ):
            for j in range(0,self.size_out ):
                self.train_values_out[j] += self.weights1[i][j]*self.train_values_h[i]
        for i in range(0,self.size_out ):
            self.train_values_out[i] += self.free_mem_out[i]
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        return self.train_values_out
        
    def learning(self, X,y, coef=0.1):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h = np.dot(X,self.weights0)
        
        self.train_values_h += self.free_mem_h
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        ##########
        self.train_values_out = np.dot(self.train_values_h,self.weights1)
        
        self.train_values_out += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        self.delta_out = y - self.train_values_out
        
        for k in range(0,self.size_out):
            self.delta_out[k] = self.delta_out[k] * coef * sigmoid(self.train_values_out[k],deriv = True)
            
        self.delta_hidden = np.dot(self.weights1,self.delta_out)
        for k in range(0,self.size_hidden):
            self.delta_hidden[k] * coef * sigmoid(self.train_values_h[k],deriv = True)
            
        self.weights1 +=  - np.dot(self.delta_out[:,None],self.train_values_h[:,None].T).T
        # c = np.dot(b[:,None],a[:,None].T)
        self.weights0 += - np.dot(self.delta_hidden[:,None],X[:,None].T).T
        return 1
        
        
    def saveP(self):
        import pickle
        with open('data.pickle', 'wb') as f:
            pickle.dump(self, f)
    def load(self):
        import pickle
        with open('data.pickle', 'rb') as f:
            self_new = pickle.load(f)
            self.size_in = self_new.size_in
            self.size_hidden = self_new.size_hidden
            self.size_out = self_new.size_out
            self.input = np.ones(size_in)
            self.weights0 = np.random.random([size_in,size_hidden])-0.5
            self.delta_hidden = np.zeros(size_hidden)
            self.free_mem_h = np.ones(size_hidden)*0.5
            self.train_values_h = np.zeros(size_hidden)####
            self.weights1 = np.random.random([size_hidden, size_out])-0.5
            self.delta_out = np.zeros(size_out)
            self.free_mem_out = np.ones(size_out)*0.5
            self.train_values_out = np.random.random(size_out)
        print self

data = pandas.read_csv('train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
from time import time

t = time()


MyNN = NN( 784,100,10)

y_temp = np.zeros(10)
print t

for i in range (1,20000):
    print i
    for j in range(0,1000):
        y_temp[y_train[i]] = 1
        MyNN.learning(X = X_train[i],y = y_temp,coef =0.1)
        y_temp[y_train[i]] = 0

#1000 acc = 0.05
#20000 acc = 0.09        
MyNN.saveP()

for i in range(20000,20200):
    print np.argmax(MyNN.predict(X_train[i])) == y_train[i]

print "time"
print time()-t