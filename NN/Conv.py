# -*- coding: utf-8 -*-
import numpy as np
import pandas

featuresMap = 15
n_matr = 3#don't change

def sigmoid(x, alfa = 1/10.0, deriv = False):
    if (deriv == False):
         return 1.0/(1.0+np.exp(-x*alfa))
    return x*(1.0-x)

class NN:
    "neiral network direct propagation(feedforward)"
    def __init__(self, size_in, size_hidden, size_out, ConvL = [1]):
        #initialization
        self.size_in = size_in
        self.size_hidden = size_hidden# example: size = np.array([100,150])
        self.size_out = size_out      # all matrix n*n
		
        #hidden layers
        if( 1 in ConvL):
		    size_hidden[0]= (np.sqrt(size_in) - n_matr+1)**2 * featuresMap
		self.edge = np.sqrt(self.size_in)
        self.weights = []
        self.delta_hidden = []
        self.bias_h = []
        self.train_values_h = []

        self.weights.append(np.random.random([size_in,self.size_hidden[0]]))
        self.delta_hidden.append( np.zeros(self.size_hidden))  #error for one neiron, and after local gragient
        self.bias_h.append(np.random.random_sample())
        self.train_values_h.append(np.zeros(self.size_hidden[0]))#last calculate out of neiron
        
        for i in range(1,len(self.size_hidden)):
            self.weights.append(np.random.random([size_hidden[i-1],self.size_hidden[i]]))
            self.delta_hidden.append( np.zeros(self.size_hidden))  #error for one neiron, and after local gragient
            self.bias_h.append(np.random.random_sample())
            self.train_values_h.append(np.zeros(self.size_hidden[i]))#last calculate out of neiron
            
        #out layer
        self.weights_out = np.random.random([self.size_hidden[-1], size_out])
        self.delta_out = np.zeros(size_out)
        self.bias_out = np.random.random_sample()
        self.train_values_out = np.zeros(size_out)
        
        self.E_out = np.zeros(size_out)# Error in last repeat for out neiral network

	def convolution(x,y,num, core):#свёртка - что мы получим в train_values на свёртке вокруг элемента (x,y) помноженному на ядро
		'''num - numeral of layer, core - matrix n_matr*n_matr '''
		#self.weights[num][[np.array([[y+1,y+1,y+1],[y,y,y],[y-1,y-1,y-1]])],np.array([[x-1,x,x+1],[x-1,x,x+1],[x-1,x,x+1]])]
		if(x==0 or x ==self.edge-1 or y == 0 or y ==self.edge-1):
		    return 0
		return np.sum(np.multiply(self.weights[num][[np.array([[y+1,y+1,y+1],[y,y,y],[y-1,y-1,y-1]])],np.array([[x-1,x,x+1],[x-1,x,x+1],[x-1,x,x+1]])],core))
		
    def predict(self,X): # X - input example
    #prediction out of NN

# x.reshape((28,28))[[np.array([[1,1,1],[2,2,2],[3,3,3]])],np.array([[1,2,3],[1,2,3],[1,2,3]])]
        for i in range(1, featuresMap+1):
			 
		
        self.train_values_h[0] = np.dot(self.weights[0].T,X)
        self.train_values_h[0] += self.bias_h[0]
        for i in range(0,self.size_hidden[0] ):
            self.train_values_h[0][i] = sigmoid(self.train_values_h[0][i])

        for j in range(1,len(self.size_hidden)):
            self.train_values_h[j] = np.dot(self.weights[j].T,self.train_values_h[j-1] )
            self.train_values_h[j] += self.bias_h[j]
            for i in range(0,self.size_hidden[j]):
                self.train_values_h[j][i] = sigmoid(self.train_values_h[j][i])
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_h[-1])
 
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])

        return self.train_values_out
        
    def learning(self, X,y, coef=0.1): # X - input example, y - expected yield, coef - coefficient of learning
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        
        self.train_values_h[0] = np.dot(self.weights[0].T,X)
        self.train_values_h[0] += self.bias_h[0]
        for i in range(0,self.size_hidden[0] ):
            self.train_values_h[0][i] = sigmoid(self.train_values_h[0][i])

        for j in range(1,len(self.size_hidden)):
            self.train_values_h[j] = np.dot(self.weights[j].T,self.train_values_h[j-1] )
            self.train_values_h[j] += self.bias_h[j]
            for i in range(0,self.size_hidden[j]):
                self.train_values_h[j][i] = sigmoid(self.train_values_h[j][i])
        
        self.train_values_out = np.dot(self.weights_out.T,self.train_values_h[-1])
 
        self.train_values_out += self.bias_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
      
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        #self.delta_out = self.delta_out  * self.train_values_out* ( 1.0 - self.train_values_out)
        self.delta_out = self.delta_out  * sigmoid(self.train_values_out, deriv = True)

        self.delta_hidden[-1] = np.dot(self.delta_out,self.weights_out.T)
#        self.delta_hidden[-1] = self.delta_hidden[-1]  * self.train_values_h[-1] * ( 1.0- self.train_values_h[-1])
        self.delta_hidden[-1] = self.delta_hidden[-1]  * sigmoid(self.train_values_h[-1], deriv = True)
        for j in range(len(self.size_hidden)-2 ,-1 ,-1):
            self.delta_hidden[j] = np.dot(self.delta_hidden[j+1],self.weights[j+1].T)
 #         self.delta_hidden[j] = self.delta_hidden[j]  * self.train_values_h[j] * ( 1.0- self.train_values_h[j])
            self.delta_hidden[j] = self.delta_hidden[j]  * sigmoid(self.train_values_h[j], deriv = True)
        self.weights_out = self.weights_out*1.0 - coef * np.dot(self.train_values_h[-1][:,None],self.delta_out[:,None].T)
        
        if (len(self.size_hidden)>1):
            for i in range(len(self.size_hidden)-1,0,-1):
                self.weights[i] = self.weights[i]*1.0 - coef * np.dot(self.train_values_h[i-1][:,None],self.delta_hidden[i][:,None].T)
        self.weights[0] = self.weights[0]*1.0 - coef * np.dot(X[:,None],self.delta_hidden[0][:,None].T)

        return 1

data = pandas.read_csv('train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
X_train = X_train.as_matrix()
X_train[X_train<100] = 0   #normalization
X_train[X_train>1] = 1
from time import time
t = time()


#MyNN = NN( 784,[100],10)
#y_temp = np.zeros(10)
#for i in np.random.randint(42000, size = 30000):
#    y_temp[y_train[i]] = 1
#    MyNN.learning(X = X_train[i],y = y_temp,coef =0.8)
#    y_temp[y_train[i]] = 0
#
#for i in np.random.randint(42000, size = 2000):
#    y_temp[y_train[i]] = 1
#    MyNN.learning(X = X_train[i],y = y_temp,coef =0.3)
#    y_temp[y_train[i]] = 0
MyNN = NN( 784,[100],10)
y_temp = np.zeros(10)
lear = 0
step_c = 0
for i in np.random.randint(42000, size = 30000):
    y_temp[y_train[i]] = 1
    MyNN.learning(X = X_train[i].reshape([28,28]),y = y_temp,coef =0.7)
    if ( np.argmax(MyNN.predict(X = X_train[i])) == y_train[i]):
        lear +=1
        if(lear == 1000):
            print "step_c=", step_c 
            break
    else:
        lear -=1
    step_c+=1
    y_temp[y_train[i]] = 0



#for i in range(50,60):
#    print MyNN.predict(X_train[i])
#    print y_train[i]    

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