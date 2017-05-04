import numpy as np
import pandas

featuresMap = 15
n_matr = 3
alfa = 1.0/10

def sigmoid(x, alfa = 1/10.0, deriv = False):
    if (deriv == False):
         return 1.0/(1.0+np.exp(-x*alfa))
    return x*(1.0-x)
def convolutional(X, core):
    out = np.zeros([26,26])
    for i in range(1,26):
        for j in range(1,26):
            out[i-1][j-1] = sigmoid (X.reshape([28,28])[i-1,j+1] * core.reshape([3,3])[0,0] + X.reshape([28,28])[i,j+1] * core.reshape([3,3])[0,2] + X.reshape([28,28])[i+1,j+1] * core.reshape([3,3])[0,2] + X.reshape([28,28])[i-1,j] * core.reshape([3,3])[1,0] + X.reshape([28,28])[i,j] * core.reshape([3,3])[1,1] + X.reshape([28,28])[i+1,j] * core.reshape([3,3])[1,2] + X.reshape([28,28])[i-1,j-1] * core.reshape([3,3])[2,0] + X.reshape([28,28])[i,j-1] * core.reshape([3,3])[2,1] + X.reshape([28,28])[i+1,j-1] * core.reshape([3,3])[2,2])
            #out[i-1][j-1] = X.reshape([28,28])[i-1,j+1]
            #out[i-1][j-1] = sigmoid (out[i-1][j-1])
    return out.reshape(676)

    
class CNN:
    "convolutional neiral network direct propagation(feedforward)"
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        
        #784, 28*28
        #676  26*26
        self.size_c = featuresMap*26*26
        self.weights_c = np.random.random([featuresMap,n_matr,n_matr])
        self.train_values_c = np.zeros([featuresMap,26,26])
        self.bias_c  = np.random.random_sample()
        self.delta_c = np.zeros([featuresMap,26,26])
        
        self.weights_out = np.random.random([featuresMap*26*26,size_out])
        self.train_values_out = np.zeros(size_out)
        self.bias_out  = np.random.random_sample()
        self.delta_out = np.zeros(size_out)
        
        self.E_out = np.zeros(size_out)# Error in last repeat for out neiral network
    
    
    def predict(self,X):
#        for i in range(0,featuresMap):
#            self.train_values_c[i] = convolutional(X,self.weights_c[i] )
        self.train_values_c = np.array([convolutional(X, elem) for elem in self.weights_c])
        self.train_values_c += self.bias_c
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_c)
        self.train_values_out += self.bias_out
        self.train_values_out = np.array([sigmoid(values) for values in self.train_values_out ])
        return self.train_values_out
        
    def learning(self, X, y, coef = 0.1):
        self.train_values_c = np.array([convolutional(X, elem) for elem in self.weights_c])
        self.train_values_c += self.bias_c
        
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_c.reshape(self.size_c))
        self.train_values_out += self.bias_out
        self.train_values_out = np.array([sigmoid(values) for values in self.train_values_out ])
        
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        self.delta_out = self.delta_out  * sigmoid(self.train_values_out, deriv = True)
        self.delta_c = np.dot(self.delta_out,self.weights_out.T).reshape([featuresMap,26,26])
        self.delta_c = self.delta_c * sigmoid(self.train_values_c, deriv = True).reshape([featuresMap,26,26])
        
        self.weights_out = self.weights_out*1.0 - coef * np.dot(self.train_values_c.reshape(self.size_c)[:,None],self.delta_out[:,None].T)
        self.weights_c = self.weights_c*1.0 - coef * np.dot(X[:,None],self.delta_c[:,None].T)
        
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
for i in np.random.randint(42000, size = 30000):
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

acc = 0.0 
num = np.zeros(10)
for i in np.random.randint(42000, size = 1000):
   if(np.argmax(MyNN.predict(X_train[i])) == y_train[i]):
    acc+=1
    num[y_train[i]]+=1
   else:
#        imgplot = plt.imsave( '%d' % i ,np.reshape(X_train[i],(28,28)))
          acc+=0.0
print "accurance:", acc/1000
print num
print "time"
print time()-t