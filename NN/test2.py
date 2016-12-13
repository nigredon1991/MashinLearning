#
import numpy as np
import pandas

def sigmoid(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))
    
    
class NN:
    "neiral network direct propagation"
    def __init__(self, size_in, size_hidden0, size_hidden1,size_hidden2, size_out):
        
        self.size_in = size_in
        self.size_hidden0 = size_hidden0
        self.size_hidden1 = size_hidden1
        self.size_hidden2 = size_hidden2
        self.size_out = size_out

        
        self.weights0 = np.random.random([size_in,size_hidden0])
        self.delta_hidden0 = np.zeros(size_hidden0)
        self.free_mem_h0 = 0.3
        self.train_values_h0 = np.zeros(size_hidden0)####

        self.weights1 = np.random.random([size_hidden0, size_hidden1])
        self.delta_hidden1 = np.zeros(size_hidden1)
        self.free_mem_h1 = 0.3
        self.train_values_h1 = np.zeros(size_hidden1)

        self.weights2 = np.random.random([size_hidden1, size_hidden2])
        self.delta_hidden2 = np.zeros(size_hidden2)
        self.free_mem_h2 = 0.5
        self.train_values_h2 = np.zeros(size_hidden2)
        
        self.weights3 = np.random.random([size_hidden2, size_out])
        self.delta_out = np.zeros(size_out)
        self.free_mem_out = 0.3
        self.train_values_out = np.zeros(size_out)
        
        self.E_out = np.zeros(size_out)

    def predict(self,X):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        
        self.train_values_h0 = np.dot(self.weights0.T,X)
        self.train_values_h0 += self.free_mem_h0
        for i in range(0,self.size_hidden0 ):
            self.train_values_h0[i] = sigmoid(self.train_values_h0[i])

        self.train_values_h1 = np.dot(self.weights1.T,self.train_values_h0)
        self.train_values_h1 += self.free_mem_h1
        for i in range(0,self.size_hidden1 ):
            self.train_values_h1[i] = sigmoid(self.train_values_h1[i])

        self.train_values_h2 = np.dot(self.weights2.T,self.train_values_h1)
        self.train_values_h2 += self.free_mem_h2
        for i in range(0,self.size_hidden2 ):
            self.train_values_h2[i] = sigmoid(self.train_values_h2[i])
        
        self.train_values_h3 = np.dot(self.weights3.T,self.train_values_h2)
        self.train_values_h3 += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])     
        
        return self.train_values_out    
        
    def learning(self, X,y, coef=0.1):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        self.train_values_h0 = np.dot(self.weights0.T,X)
        self.train_values_h0 += self.free_mem_h0
        for i in range(0,self.size_hidden0 ):
            self.train_values_h0[i] = sigmoid(self.train_values_h0[i])

        self.train_values_h1 = np.dot(self.weights1.T,self.train_values_h0)
        self.train_values_h1 += self.free_mem_h1
        for i in range(0,self.size_hidden1 ):
            self.train_values_h1[i] = sigmoid(self.train_values_h1[i])

        self.train_values_h2 = np.dot(self.weights2.T,self.train_values_h1)
        self.train_values_h2 += self.free_mem_h2
        for i in range(0,self.size_hidden2 ):
            self.train_values_h2[i] = sigmoid(self.train_values_h2[i])
        
        self.train_values_h3 = np.dot(self.weights3.T,self.train_values_h2)
        self.train_values_h3 += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])            
        
        
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        print self.E_out
        self.delta_out = self.delta_out * self.train_values_out* ( 0.999999 - self.train_values_out)

        self.delta_hidden2 = np.dot(self.delta_out,self.weights3.T)
        self.delta_hidden2 = self.delta_hidden2  * self.train_values_h2 * ( 0.999999- self.train_values_h2)

        self.delta_hidden1 = np.dot(self.delta_hidden2,self.weights2.T)
        self.delta_hidden1 = self.delta_hidden1  * self.train_values_h1 * ( 0.999999- self.train_values_h1)

        self.delta_hidden0 = np.dot(self.delta_hidden1,self.weights1.T)
        self.delta_hidden0 = self.delta_hidden0  * self.train_values_h0 * ( 0.999999- self.train_values_h0)
        #print self.delta_out
        self.weights3 -= coef * np.dot(self.train_values_h2[:,None],self.delta_out[:,None].T)
        self.weights2 -= coef * np.dot(self.train_values_h1[:,None],self.delta_hidden2[:,None].T)
        self.weights1 -= coef * np.dot(self.train_values_h0[:,None],self.delta_hidden1[:,None].T)
        self.weights0 -= coef * np.dot(X[:,None],                   self.delta_hidden0[:,None].T)
  
        return  self.E_out
        
    def saveP(self):
        import pickle
        with open('data.pickle', 'wb') as f:
            pickle.dump(self, f)
    def load(self):
        import pickle
        with open('data.pickle', 'rb') as f:
            self_new = pickle.load(f)
            self.size_in = self_new.size_in
            self.size_hidden0 = self_new.size_hidden0
            self.size_hidden1 = self_new.size_hidden1
            self.size_hidden2 = self_new.size_hidden2
            self.size_out = self_new.size_out
            self.weights0 = self_new.weights0
            self.delta_hidden0 =self_new.delta_hidden0
            self.free_mem_h0 = self_new.free_mem_h0
            self.train_values_h0 = self_new.train_values_h0
            self.weights1 = self_new.weights1
            self.delta_hidden1 = self_new.delta_hidden1
            self.free_mem_h1 = self_new.free_mem_h1
            self.train_values_h1 = self_new.train_values_h1
            self.weights2 = self_new.weights2
            self.delta_hidden2 = self_new.delta_hidden2
            self.free_mem_h2 = self_new.free_mem_h2
            self.train_values_h2 = self_new.train_values_h2
            self.weights3 = self_new.weights3
            self.delta_out = self_new.delta_out
            self.free_mem_out = self_new.free_mem_out
            self.train_values_out = self_new.train_values_out
            self.E_out = self_new.E_out
        print self

#data = pandas.read_csv('train.csv')
#y_train = data["label"] # 42000
#
#X_train = data.drop("label", axis = 1) # 42000 * 784
#X_train = X_train.as_matrix()
#from sklearn import preprocessing
##min_max_scaler = preprocessing.MinMaxScaler()
##X_train = min_max_scaler.fit_transform(X_train)
##from time import time
#X_train[X_train>0] = 1
#
from time import time
t = time()
#MyNN = NN( 784,150,150,50,10)

MyNN = NN( 2,10,5,5,1)
X = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([1,0,0,1])
for j in range(0,100):
    for i in np.random.randint(4,size = 100):
        MyNN.learning(X=X[i],y=y[i], coef = 0.5)

for i in range(0,4):
    print MyNN.predict(X=X[i])
#

#y_temp = np.zeros(10)
#print t
#
#for i in np.random.randint(42000,size = 20000):
#    #print i
#    y_temp[y_train[i]] = 0.99
#    if (MyNN.learning(X = X_train[i],y = y_temp,coef =1 ) < 0.001 ):
#        break
#    y_temp[y_train[i]] = 0.01
    #print MyNN.predict(X_train[i])
    #print y_train[i]
#print MyNN.weights0[10]
#for i in np.random.randint(20000,size = 30000):
#    #print i
#    y_temp[y_train[i]] = 1
#   MyNN.learning(X = X_train[i],y = y_temp,coef =0.1)
#    y_temp[y_train[i]] = 0

#for i in range(50,60):
#    print MyNN.predict(X_train[i])
#    print y_train[i]    

#    print MyNN.predict(X_train[i])
#1000 acc = 0.1
        
#MyNN.saveP()


#for i in range(20000,20010):
#     print MyNN.predict(X_train[i])
#     print y_train[i]
#
#y_temp[y_train[10]] = 1
#print MyNN.learning(X = X_train[10],y = y_temp,coef =0.5 )
#y_temp[y_train[10]] = 0
#for i in range(20000,20200):
#    print np.argmax(MyNN.predict(X_train[i])) == y_train[i]

print "time"
print time()-t