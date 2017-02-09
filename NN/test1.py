#
import numpy as np
import pandas

def sigmoid(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

class NN:
    "neiral network direct propagation"
    def __init__(self, size_in, size_hidden, size_out):
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.input = np.ones(size_in)
        self.weights0 = np.random.random([size_in,size_hidden])-0.5
        self.delta_hidden = np.zeros(size_hidden)
        self.free_mem_h = 0#np.random.random_sample()
        self.train_values_h = np.zeros(size_hidden)####
        self.weights1 = np.random.random([size_hidden, size_out])-0.5
        self.delta_out = np.zeros(size_out)
        self.free_mem_out = 0#np.random.random_sample()
        self.train_values_out = np.zeros(size_out)
        self.E_out = np.zeros(size_out)

    def predict(self,X):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h = np.dot(self.weights0.T,X)
        self.train_values_h += self.free_mem_h
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        
        self.train_values_out = np.dot(self.weights1.T,self.train_values_h)
        self.train_values_out += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        
        return self.train_values_out
        
    def learning(self, X,y, coef=0.1):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h = np.dot(self.weights0.T,X)
        self.train_values_h += self.free_mem_h
        #self.train_values_h[self.train_values_h>0.9] = 0.9
        #self.train_values_h[self.train_values_h<0.1] = 0.1
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        
        self.train_values_out = np.dot(self.weights1.T,self.train_values_h)
        self.train_values_out += self.free_mem_out
        #self.train_values_out[self.train_values_out>0.9] = 0.9
        #self.train_values_out[self.train_values_out<0.1] = 0.1

        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
      
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        print self.E_out
        self.delta_out = self.delta_out  * self.train_values_out* ( 1 - self.train_values_out)
        
        self.delta_hidden = np.dot(self.delta_out,self.weights1.T)
       # print  self.delta_hidden
        self.delta_hidden = self.delta_hidden  * self.train_values_h * ( 1- self.train_values_h)
        #print  self.delta_hidden
        self.weights1 = self.weights1*0.9 -  coef * np.dot(self.train_values_h[:,None],self.delta_out[:,None].T)
        #MyNN.weights1[MyNN.weights1<0] = 0.000001
        #print np.dot(self.train_values_h[:,None],self.delta_out[:,None].T)
        self.weights0 = self.weights0 * 0.99 - coef * np.dot(X[:,None],self.delta_hidden[:,None].T)
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
            self.input = self_new.input
            self.weights0 = self_new.weights0
            self.delta_hidden = self_new.delta_hidden
            self.free_mem_h = self_new.free_mem_h
            self.train_values_h = self_new.train_values_h
            self.weights1 = self_new.weights1
            self.delta_out = self_new.delta_out
            self.free_mem_out = self_new.free_mem_out
            self.train_values_out = self_new.train_values_out
        print self

data = pandas.read_csv('NN/train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
X_train = X_train/512
from time import time

t = time()


MyNN = NN( 784,100,10)

y_temp = np.zeros(10)+0.0001
print t

for i in np.random.randint(5000,size = 10000):
   # print i
    for j in range(0,2):
        y_temp[y_train[i]] = 0.9999
        MyNN.learning(X = X_train[i],y = y_temp,coef =1)
        y_temp[y_train[i]] = 0
    print MyNN.predict(X_train[i])
    print y_train[i]

for i in range(50,60):
    print MyNN.predict(X_train[i])
    print y_train[i]    
#for i in range (1,20000):
#   print i,"__________"
#    for j in range(0,10000):
#        y_temp[y_train[i]] = 0.99
#        MyNN.learning(X = X_train[i],y = y_temp,coef =0.5)
#        y_temp[y_train[i]] = 0.05
    
#    print MyNN.predict(X_train[i])
#1000 acc = 0.1
        
#MyNN.saveP()

for i in range(20000,20200):
    print np.argmax(MyNN.predict(X_train[i])) == y_train[i]

print "time"
print time()-t