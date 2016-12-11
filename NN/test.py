#
import numpy as np
import pandas

def sigmoid(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return np.exp(-x)/(1+np.exp(-x)**2)

    
class NN:
    "neiral network direct propagation"
    def __init__(self, size_in, size_hidden, size_out):
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.input = np.ones(size_in)
        self.weights0 = np.random.random([size_in,size_hidden])
        self.delta_hidden = np.zeros(size_hidden)
        self.free_mem_h = np.random.random_sample()
        self.train_values_h = np.zeros(size_hidden)####
        self.weights1 = np.random.random([size_hidden, size_out])
        self.delta_out = np.zeros(size_out)
        self.free_mem_out = np.random.random_sample()
        self.train_values_out = np.zeros(size_out)
        self.E_out = np.zeros(size_out)

    def predict(self,X):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h = np.dot(self.weights0,X)
        self.train_values_h += self.free_mem_h
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        ##########
        self.train_values_out = np.dot(self.weights1,self.train_values_h)
        self.train_values_out += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        return self.train_values_out
        
    def learning(self, X,y, coef=0.1):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h = np.dot(self.weights0,X)
        self.train_values_h += self.free_mem_h
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        ##########
        self.train_values_out = np.dot(self.weights1,self.train_values_h)
        self.train_values_out += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
      
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        #print self.train_values_h
        self.delta_out = self.delta_out  * self.train_values_out* ( 1 - self.train_values_out)
        
        self.delta_hidden = np.dot(self.delta_out,self.weights1)
       # print  self.delta_hidden
        self.delta_hidden = self.delta_hidden  * self.train_values_h * ( 1- self.train_values_h)
        #print  self.delta_hidden
        self.weights1 -= coef * np.dot(self.delta_out[:,None],self.train_values_h[:,None].T)
        #print np.dot(self.train_values_h[:,None],self.delta_out[:,None].T)
        self.weights0 -= coef * np.dot(self.delta_hidden[:,None],X[:,None].T)
        return 1
        

from time import time

t = time()


MyNN = NN( 2,2,2)
X_train = np.array([0.05, 0.1])
y_temp = np.array([0.01,0.99])
MyNN.free_mem_h = 0.35
MyNN.free_mem_out = 0.6
MyNN.weights0 = [[0.15,0.2],[0.25,0.3]]
MyNN.weights1 = [[0.4,0.45],[0.5,0.55]]

#print t

MyNN.learning(X = X_train,y = y_temp,coef =0.5)
#MyNN.learning(X = X_train,y = y_temp,coef =0.5)
#print MyNN.predict(X_train)
#print 'weights0\n', MyNN.weights0
#print 'weights1\n', MyNN.weights1
for i in range(0,10000):
    MyNN.learning(X = X_train,y = y_temp,coef =0.5)
print MyNN.E_out
print MyNN.predict(X_train)
print "time"
print time()-t