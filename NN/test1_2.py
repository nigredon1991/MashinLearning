#
import numpy as np
import pandas

def sigmoid(x,deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

class NN:
    "neiral network direct propagation"
    def __init__(self, size_in, size_hidden0, size_hidden1, size_out):
        
        self.size_in = size_in
        self.size_hidden0 = size_hidden0
        self.size_hidden1 = size_hidden1
        self.size_out = size_out

        self.weights0 = np.random.random([size_in,size_hidden0])
#        N = size_in/size_hidden0
#        for i in range(size_in):
#            for j in range(size_hidden0):
#                if(j<N*i):
#                    self.weights0[i][j]=0
#                if(j>(N+N-1)*i):
#                    self.weights0[i][j]=0 
        self.delta_hidden0 = np.zeros(size_hidden0)
        self.free_mem_h0 = np.random.random_sample()
        self.train_values_h0 = np.zeros(size_hidden0)####

        self.weights1 = np.random.random([size_hidden0, size_hidden1])
        self.delta_hidden1 = np.zeros(size_hidden1)
        self.free_mem_h1 = np.random.random_sample()
        self.train_values_h1 = np.zeros(size_hidden1)

        self.weights2 = np.random.random([size_hidden1, size_out])
        self.delta_out = np.zeros(size_out)
        self.free_mem_out = np.random.random_sample()
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
            
        self.train_values_out = np.dot(self.weights2.T,self.train_values_h1)
        self.train_values_out += self.free_mem_out
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
            
        self.train_values_out = np.dot(self.weights2.T,self.train_values_h1)
        self.train_values_out += self.free_mem_out
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        
        self.delta_out = -(y - self.train_values_out)
        self.E_out = sum(self.delta_out*self.delta_out/2)
        print self.E_out
        #print y
        #print self.train_values_out
        self.delta_out = self.delta_out  * self.train_values_out* ( 1 - self.train_values_out)
        
        self.delta_hidden1 = np.dot(self.delta_out,self.weights2.T)
        self.delta_hidden1 = self.delta_hidden1  * self.train_values_h1 * ( 1- self.train_values_h1)

        self.delta_hidden0 = np.dot(self.delta_hidden1,self.weights1.T)
        self.delta_hidden0 = self.delta_hidden0  * self.train_values_h0 * ( 1- self.train_values_h0)
        #print self.delta_out
        self.weights2 -= coef * np.dot(self.train_values_h1[:,None],self.delta_out[:,None].T)
        self.weights1 -= coef * np.dot(self.train_values_h0[:,None],self.delta_hidden1[:,None].T)
        self.weights0 -= coef * np.dot(X[:,None],                   self.delta_hidden0[:,None].T)
                  
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
            self.size_hidden0 = self_new.size_hidden0
            self.size_out = self_new.size_out
            self.weights0 = self_new.weights0
            self.delta_hidden0 = self_new.delta_hidden0
            self.free_mem_h0 = self_new.free_mem_h0
            self.train_values_h0 = self_new.train_values_h0
            self.weights1 = self_new.weights1
            self.delta_out = self_new.delta_out
            self.free_mem_out = self_new.free_mem_out
            self.train_values_out = self_new.train_values_out
        print self

#data = pandas.read_csv('train.csv')
#y_train = data["label"] # 42000
#
#X_train = data.drop("label", axis = 1) # 42000 * 784
#X_train = X_train.as_matrix()
##from sklearn import preprocessing
##min_max_scaler = preprocessing.MinMaxScaler()
##X_train = min_max_scaler.fit_transform(X_train)
#X_train = X_train/255
#from time import time
#t = time()

#MyNN = NN( 784,98,15,10)
#y_temp = np.zeros(10)
#print t
#
#for i in np.random.randint(20000,size = 10000):
#    #print i
#    y_temp[y_train[i]] = 1
#    MyNN.learning(X = X_train[i],y = y_temp,coef =1)
#    y_temp[y_train[i]] = 0
#    #print MyNN.predict(X_train[i])
#   #print y_train[i]
#
#for i in range(50,60):
#    print MyNN.predict(X_train[i])
#    print y_train[i] 
#   

#from time import time
#t = time()
#MyNN = NN( 1,5,5,1)
#X = np.array([[0,0],[1,0],[0,1],[1,1]])
#y = np.array([0,1,1,0])
#
#for i in np.random.randint(4,size = 10000):
#    MyNN.learning(X=X[i],y=y[i], coef = 1)
#
#for i in range(0,4):
#    print MyNN.predict(X=X[i])

from time import time
t = time()
MyNN = NN( 2,3,3,1)
X = np.zeros((100,2))
for i in range(1,100):
    X[i][0] = 1.0/i
y = np.zeros(100)
for i in range(100):
    y[i] = np.sin(X[i][0])

for i in np.random.randint(100,size = 100000):
    MyNN.learning(X=X[i],y=y[i], coef = 1)
    
for i in np.random.randint(100,size = 4):
    print MyNN.predict(X=X[i])
    print y[i]


#for i in range (1,100):
#    print i
#    for j in range(0,500):
#        y_temp[y_train[i]] = 0.9999
#        MyNN.learning(X = X_train[i],y = y_temp,coef =0.5)
#        y_temp[y_train[i]] = 0
#    print MyNN.predict(X_train[i])
#    print y_train[i]
   
#for i in range (1,20000):
#   print i,"__________"
#    for j in range(0,10000):
#        y_temp[y_train[i]] = 0.99
#        MyNN.learning(X = X_train[i],y = y_temp,coef =0.5)
#        y_temp[y_train[i]] = 0.05
    
#    print MyNN.predict(X_train[i])
#1000 acc = 0.1
        
#MyNN.saveP()

#for i in range(20000,20200):
#    print MyNN.predict(X_train[i])
#    print y_train[i]
#
#for i in range(20000,20200):
#    print np.argmax(MyNN.predict(X_train[i])) == y_train[i]

print "time"
print time()-t