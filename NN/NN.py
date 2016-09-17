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
            return f(x)*(1-f(x))
        return 1/(1+np.exp(-x))

    def __init__(self, numberW = 1,func = sigmoid):
        self.Weights = np.random.random((numberW,1)) #Веса входных синапсов
        self.func = func                             # Активационная функция
        self.train_value = 0                         # Значение выданное при последнем обучении
    def predict(self,X):
        self.train_value = self.func(np.dot(X,self.Weights))

    def learning(self,X, real_value, coef_learning = 0.1):
        delta = real_value - self.train_value
        self.Weights = self.Weights + coef_learning * delta * self.func(self.train_value,True) * X


" x = N1(10)"
# x.predict(np.array([1,1,1,1,1,1,1,1,1,1]))