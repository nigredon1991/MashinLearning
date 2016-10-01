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
data = pandas.read_csv('NN/train.csv')
y_train = data["label"] # 42000

X_train = data.drop("label", axis = 1) # 42000 * 784
#возьмём нейроны по 98 входов: 8*98 = 784
#будет ещё выходной нейрон с 8 входами




class N1:
    "neiron with N inputs"


    def sigmoid(x,deriv=False):
        if(deriv==True):
            return (x*(1-x))
        return 1/(1+np.exp(-x))

    def __init__(self, numberW = 1,func = sigmoid):
        self.Weights = np.random.random((numberW,)) #Веса входных синапсов
        self.func = func                             # Активационная функция
        self.train_value = 0                         # Значение выданное при последнем обучении
    def predict(self,X):
        self.train_value = self.func(np.dot(X,self.Weights))
        return self.train_value

    def learning(self,X, real_value, coef_learning = 0.1):
        delta = real_value - self.train_value
        self.Weights = self.Weights + coef_learning * delta * self.func(self.train_value,True) * X


#X = np.array([1,1,1,1,1, 1,1,1,1,1])
#neir = N1(10)
#neir.predict(X)
#neir.learning(X,10)
#neir.predict(X)


# 42000 * 784
#возьмём нейроны по 98 входов: 8*98 = 784
#будут ещё выходные нейроны с 8 входами

NN1 = [N1(98),N1(98),N1(98),N1(98),N1(98),N1(98),N1(98),N1(98)]
NN2 = [N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8),N1(8)]
X2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
a = 0
for i in range(1,42000):
    for j in range(1,8):
        for k in range(1,98):
            X2[j-1]= NN1[j].predict(X_train[X_train.columns[(k*j-1):(k*j)]])
    for j in range(0,9):
        a = NN2[j].predict(X2)
        NN2[j].learning(X2,lambda x: 1 if(y_train[i] == j) else 0)
