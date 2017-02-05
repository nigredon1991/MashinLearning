#
import numpy as np
import pandas

# Сигмоида
def f(x):
    return x

def nonlin(x,deriv=False):
    if(deriv==True):
        return f(x)*(1-f(x))
    return 1/(1+np.exp(-x))

# набор входных данных
#data = pandas.read_csv('NN/train.csv')
#y_train = data["label"] # 42000

#X_train = data.drop("label", axis = 1) # 42000 * 784
#возьмём нейроны по 98 входов: 8*98 = 784
#будет ещё выходной нейрон с 8 входами



def MyMod(x):
    if(x<0):
        return -x
    else:
        return x

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
        self.free_mem_h = np.random.random(size_hidden)
        self.train_values_h = np.zeros(size_hidden)####
        self.weights1 = np.random.random([size_hidden, size_out])-0.5

        self.delta_out = np.zeros(size_out)
        self.free_mem_out = np.random.random(size_out)
        self.train_values_out = np.zeros(size_out)

    def predict(self,X):
        if(len(X)!= self.size_in):
            print("Bad len input")
            return 0
        ######Проход скрытого слоя
        for i in range(0,self.size_in ):
            for j in range(0,self.size_hidden ):
                self.train_values_h[j] += self.weights0[i][j]*X[i]
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] += self.free_mem_h[i]
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        ##########Проход выходного слоя
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
        ######Проход скрытого слоя
        for i in range(0,self.size_in ):
            for j in range(0,self.size_hidden ):
                self.train_values_h[j] += self.weights0[i][j]*X[i]
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] += self.free_mem_h[i]
        for i in range(0,self.size_hidden ):
            self.train_values_h[i] = sigmoid(self.train_values_h[i])
        ##########Проход выходного слоя
        for i in range(0,self.size_hidden ):
            for j in range(0,self.size_out ):
                self.train_values_out[j] += self.weights1[i][j]*self.train_values_h[i]
        for i in range(0,self.size_out ):
            self.train_values_out[i] += self.free_mem_out[i]
        for i in range(0,self.size_out ):
            self.train_values_out[i] = sigmoid(self.train_values_out[i])
        ###########Вычисление ошибки и обучение сети

        # if(self.train_values_out[np.argmax(y)] != 1):
        #     k = np.argmax(y)
        #     delta = 1 - self.train_values_out[k]
        #     for i in range(0,self.size_hidden ):
        #         self.weights1[i][k] = self.weights1[i][k] + delta * coef * self.train_values_out[k] * sigmoid(self.train_values_out[k],deriv = True)
        #         self.free_mem_out[i] = self.free_mem_out[i] + delta * coef * self.train_values_out[k] * sigmoid(self.train_values_out[k],deriv = True)
        #     for i in range(0,self.size_in ):
        #         for j in range(0,self.size_hidden ):
        #             self.weights0[i][j] += delta *self.weights1[j][k] * coef * self.train_values_h[j] * sigmoid(self.train_values_h[j], deriv = True)
        #             self.free_mem_h[j] += delta *self.weights1[j][k] * coef * self.train_values_h[j] * sigmoid(self.train_values_h[j], deriv = True)
        self.delta_out = y - self.train_values_out
        for i in range(0,self.size_hidden):
            for j in range(0,self.size_out):
                self.delta_hidden[i] += self.delta_out[j] * self.weights1[i][j]
        for i in range(0,self.size_hidden):
            for k in range(0,self.size_out):
                self.weights1[i][k] = self.weights1[i][k] + self.delta_out[k] * coef * self.train_values_out[k] * sigmoid(self.train_values_out[k],deriv = True)
                self.free_mem_out[i] = self.free_mem_out[i] + self.delta_out[k] * coef * self.train_values_out[k] * sigmoid(self.train_values_out[k],deriv = True)

        for i in range(0,self.size_in):
            for k in range(0,self.size_hidden):
                self.weights0[i][k] = self.weights0[i][k] + self.delta_hidden[k] * coef * self.train_values_h[k] * sigmoid(self.train_values_h[k],deriv = True)
                self.free_mem_h[i] = self.free_mem_h[i] + self.delta_hidden[k] * coef * self.train_values_h[k] * sigmoid(self.train_values_h[k],deriv = True)
        print self.delta_hidden
        self.delta_hidden = np.zeros(self.size_hidden)
        return 1


X =np.array( [[0,0],
    [1,0],
    [0,1],
    [1,1]])

y =np.array( [[1,0],[0,1],[0,1],[1,0]])

MyNN = NN( 2,2,2)

for j in range (1,800):
    for i in range(0,4):
        MyNN.learning(X = X[i],y = y[i],coef = 1.0/j)

for i in range(0,4):
    print MyNN.predict(X[i])