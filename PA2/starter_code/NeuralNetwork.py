import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    #Enter implementation here
    pass
    
def forwardPropagation(x, weights):
    #Enter implementation here
    pass

def errorPerSample(X,y_n):
    #Enter implementation here
    pass

def backPropagation(X,y_n,s,weights):
    #Enter implementation here
    pass

def updateWeights(weights,g,alpha):
    #Enter implementation here
    pass

def activation(s):
    #Enter implementation here
    return s if s >= 0 else 0

def derivativeActivation(s):
    #Enter implementation here
    return 1 if s >= 0 else 0

def outputf(s):
    #Enter implementation here
    return 1 / (1 + np.exp(-1*s))

def derivativeOutput(s):
    #Enter implementation here
    return np.exp(-1*s) / ((1 + np.exp(-1*s)) ** 2)

def errorf(x_L,y):
    #Enter implementation here
    if y == 1:
        return -1 * np.log(x_L)
    else:
        return -1 * np.log(1 - x_L)

def derivativeError(x_L,y):
    #Enter implementation here
    if y == 1:
        return -1 / x_L
    else:
        return 1 / (1 - x_L)

def pred(x_n,weights):
    #Enter implementation here
    pass
    
def confMatrix(X_train,y_train,w):
    #Enter implementation here
    pass

def plotErr(e,epochs):
    #Enter implementation here
    pass
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Enter implementation here
    pass

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test_Part1()
