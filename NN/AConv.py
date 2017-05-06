import numpy as np
import pandas
from  itertools import product 
#[(i,j) for i,j in iter.permutations(range(10),2)]
featuresMap = 8
n_matr = 3
alfa = 1.0/100.0

def sigmoid(x, alfa = 1/10.0, deriv = False):
    if (deriv == False):
         return 1.0/(1.0+np.exp(-x*alfa))
    return x*(1.0-x)
        #784, 28*28
        #676  26*26
def convolutional(X, core):
    out = np.zeros(676)
    #for i in range(1,26):
    #    for j in range(1,26):
    #        out[i-1][j-1] = sigmoid(X [i-1,j+1] * core[0,0] + X [i,j+1] * core[0,2] + X [i+1,j+1] * core[0,2] + X [i-1,j] * core[1,0] + X [i,j] * core[1,1] + X [i+1,j] * core[1,2] + X [i-1,j-1] * core[2,0] + X [i,j-1] * core[2,1] + X [i+1,j-1] * core[2,2])
    out = np.array([sigmoid (X[i-1,j+1] * core[0,0] + X[i,j+1] * core[0,1] + X[i+1,j+1] * core[0,2] + X[i-1,j] * core[1,0] + X[i,j] * core[1,1] + X[i+1,j] * core[1,2] + X[i-1,j-1] * core[2,0] + X[i,j-1] * core[2,1] + X[i+1,j-1] * core[2,2]) for i,j in product(range(1,27),repeat = 2) ])
    return out

def convolutional_back(X,k, low = False):
    if (low== True):
        i = k[0]-1
        j = k[1]-1
    else:
        i = k[0]+1
        j = k[1]+1
    #print i,j
    #print  np.array (X[i-1,j+1] )
    return np.array([ [X[i-1,j+1] ,  X[i,j+1] , X[i+1,j+1] ] , [X[i-1,j], X[i,j] , X[i+1,j] ] ,  [X[i-1,j-1], X[i,j-1] ,X[i+1,j-1] ]])
    
    
class CNN:
    "convolutional neiral network direct propagation(feedforward)"
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.size_c = featuresMap*26*26
        self.weights_c = np.random.random([featuresMap,n_matr,n_matr])-0.5
        self.train_values_c = np.zeros([featuresMap,26,26])
        self.bias_c  = np.random.random_sample()
        self.delta_c = np.zeros([featuresMap,26,26])
        #self.delta_conv = np.zeros(featuresMap)

       # self.weights_p = np.random.random([featuresMap,n_matr,n_matr])
      #  self.train_values_p = np.zeros([featuresMap,26,26])
      #  self.bias_p  = np.random.random_sample()
      #  self.delta_p = np.zeros([featuresMap,26,26])

        
        self.weights_out = np.random.random([featuresMap*26*26,size_out])-0.5
        #self.out_drop = np.random.randint(2, size = (featuresMap*26*26, size_out))
        #self.weights_out = self.weights_out * self.out_drop
        self.train_values_out = np.zeros(size_out)
        self.bias_out  = np.random.random_sample()
        self.delta_out = np.zeros(size_out)
        self.E_out = np.zeros(size_out)# Error in last repeat for out neiral network
    
    def predict(self,X):
        self.train_values_c = np.array([convolutional(X.reshape([28,28]), elem) for elem in self.weights_c])
        #self.train_values_c += self.bias_c        
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_c.reshape(self.size_c))
        #self.train_values_out += self.bias_out
        self.train_values_out = np.array([sigmoid(values) for values in self.train_values_out ])
        return self.train_values_out
        
    def learning(self, X, y, coef = 0.1):
        self.train_values_c = np.array([convolutional(X.reshape([28,28]), elem).reshape(26,26) for elem in self.weights_c])
        #self.train_values_c += self.bias_c
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_c.reshape(self.size_c))
        #self.train_values_out += self.bias_out
        self.train_values_out = np.array([sigmoid(values) for values in self.train_values_out ])
        
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        self.delta_out = self.delta_out  * sigmoid(self.train_values_out, deriv = True)
        
        self.delta_c = np.dot(self.delta_out,self.weights_out.T).reshape([featuresMap,26,26])
        self.delta_c = self.delta_c * sigmoid(self.train_values_c, deriv = True)
        #self.delta_c = self.delta_c * X.reshape([28,28])
        #self.delta_conv = np.array([ np.argmax(elem) for elem in self.delta_c])
    
        self.weights_out = self.weights_out*1.0 - coef * np.dot(self.train_values_c.reshape(self.size_c)[:,None],self.delta_out[:,None].T)
        #self.weights_out = self.weights_out * self.out_drop# convolutional_back(delta.reshape([26,26]), np.unravel_index(delta.argmax(), delta.shape), low = True)
        self.weights_c = np.array([weights_old - coef * convolutional_back(X.reshape([28,28]), np.unravel_index(delta.argmax(), delta.shape))* delta.argmax() for delta,weights_old in zip (self.delta_c, self.weights_c)])

data = pandas.read_csv('train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
X_train[X_train<100] = 0   #normalization
X_train[X_train>1] = 1
from time import time
t = time()

MyNN = CNN(784,10)

y_temp = np.zeros(10)
lear = 0
step_c = 0
for i in np.random.randint(42000, size = 1000):
    y_temp[y_train[i]] = 1
    MyNN.learning(X = X_train[i],y = y_temp,coef =0.7)
    if ( np.argmax(MyNN.predict(X = X_train[i])) == y_train[i]):
        lear +=1
        if(lear == 1000):
            print "step_c=", step_c 
            break
    else:
        lear -=1
    step_c+=1
    y_temp[y_train[i]] = 0



for i in range(50,60):
    print MyNN.predict(X_train[i])
    print y_train[i]    

#import matplotlib.pyplot as plt
#Verification
acc = 0.0 
num = np.zeros(10)
sum = np.zeros(10)
for i in np.random.randint(42000, size = 100):
    sum[y_train[i]]+= 1
    if(np.argmax(MyNN.predict(X_train[i])) == y_train[i]):
        acc+=1
        num[y_train[i]]+=1

print "accurance:", acc/100
print num
print sum
print "time"
print time()-t
