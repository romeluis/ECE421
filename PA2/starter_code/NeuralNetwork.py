import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    #Enter implementation here
    # Initialize the epoch errors
    err = np.zeros((epochs, 1))
    
    # Initialize the architecture
    N, d = X_train.shape
    X0 = np.ones((N, 1))
    X_train = np.hstack((X0, X_train))
    d = d + 1
    L = len(hidden_layer_sizes)
    L = L + 2
    
    #Initializing the weights for input layer
    weight_layer = np.random.normal(0, 0.1, (d, hidden_layer_sizes[0])) #np.ones((d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer) #append(0.1*weight_layer)
    
    #Initializing the weights for hidden layers
    for l in range(L - 3):
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l] + 1, hidden_layer_sizes[l + 1])) 
        weights.append(weight_layer) 

    #Initializing the weights for output layers
    weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l + 1] + 1,1)) 
    weights.append(weight_layer) 
    
    for e in range(epochs):
        print(e)
        choiceArray = np.arange(0, N)
        np.random.shuffle(choiceArray)
        errN = 0
        for n in range(N): 
            index = choiceArray[n]
            x = np.transpose(X_train[index])
            #TODO: Model Update: Forward Propagation, Backpropagation
            # update the weight and calculate the error
            retX, retS = forwardPropagation(x, weights)
            g = backPropagation(retX, y_train[index], retS, weights)
            errN += errorPerSample(retX, y_train[index])
            
            weights = updateWeights(weights, g, alpha)
             
        err[e] = errN / N 
    return err, weights
    
def forwardPropagation(x, weights):
    #Enter implementation here
    l = len(weights) + 1
    currX = x
    
    retS = []
    
    retX = []
    retX.append(currX)

    for i in range(l - 1):

        currS = np.dot(retX[-1], weights[i]) # TODO: Dot product between the layer and the weight matrix
        retS.append(currS)
        
        currX = currS
        
        if i != len(weights) - 1:
            for j in range(len(currS)):
                currX[j] = activation(currS[j]) # TODO: Apply the activation
            
            currX = np.hstack(([1], currX))
            
        else:
            currX = outputf(currS) #TODO: Apply the output activation
            
        retX.append(currX)
        
    return retX, retS

def errorPerSample(X, y_n):
    #Enter implementation here
    return errorf(X[-1][0], y_n)

def backPropagation(X, y_n, s, weights):
    #Enter implementation here
    l = len(X)
    delL = list()

    delL.insert(0, derivativeError(X[l - 1], y_n) * derivativeOutput(s[l - 2]))
    curr = 0
    
    for i in range(len(X) - 2, 0, -1):
        
        delNextLayer = delL[curr]
        WeightsNextLayer = weights[i]
        sCurrLayer = s[i - 1]
        
        delN = np.zeros((len(s[i - 1]), 1))
        
        for j in range(len(s[i - 1])):
            for k in range(len(s[i])):
                
                #TODO: calculate delta at node j
                delN[j] = delN[j] + derivativeActivation(sCurrLayer[k]) * (np.dot(np.transpose(WeightsNextLayer[j]), delNextLayer)) # Fill in the rest
        
        delL.insert(0, delN)
    
    
    g = []
    for i in range(len(delL)):
        
        rows, cols= weights[i].shape
        
        gL = np.zeros((rows,cols))
        
        currX = X[i]
        currdelL = delL[i]
        
        for j in range(rows):
            for k in range(cols):
                
                #TODO: Calculate the gradient using currX and currdelL
                # print(currX)
                # print(currdelL)
                cx = currX[j]
                cd = currdelL[j] if j < len(currdelL) else 0
                gL[j, k] = cx * cd # Fill in here
                
        g.append(gL)
        
    return g

def updateWeights(weights, g, alpha):
    #Enter implementation here
    nW = []
    for i in range(len(weights)):
        
        rows, cols = weights[i].shape
        currWeight = weights[i]
        currG = g[i]
        
        for j in range(rows):
            for k in range(cols):
                
                #TODO: Gradient Descent Update
                currWeight[j, k] = currWeight[j, k] - alpha * currG[j, k] # Fill in here

        nW.append(currWeight)
        
    return nW

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

def errorf(x_L, y):
    #Enter implementation here
    if y == 1:
        return -1 * np.log(x_L)
    else:
        return -1 * np.log(1 - x_L)

def derivativeError(x_L, y):
    #Enter implementation here
    if y == 1:
        return -1 / x_L
    else:
        return 1 / (1 - x_L)

def pred(x_n, weights):
    #Enter implementation here
    # TODO: prediction using the forwardPropagation function
    retX, retS = forwardPropagation(x_n, weights) # Fill in here
    l=len(retX)

    # Return -1 if probability lesser than 0.5
    # Else return 1
    if retX[l - 1] < 0.5:
        return -1
    else:
        return 1
    
def confMatrix(X_train, y_train, w):
    #Enter implementation here
    ones_to_be_added = np.ones(len(X_train))
    X = np.hstack((np.atleast_2d(ones_to_be_added).T, X_train))
    
    matrix = [[0, 0], [0, 0]]
    
    for i, x_i in enumerate(X):
        prediction = pred(x_i, w)
        if y_train[i] == -1:
            if prediction == -1:
                matrix[0][0] += 1
            else:
                matrix[0][1] += 1
        elif y_train[i] == 1:
            if prediction == -1:
                matrix[1][1] += 1
            else:
                matrix[1][0] += 1
                
    return matrix

def plotErr(e,epochs):
    #Enter implementation here
    plt.plot(range(epochs), e, linewidth=2.0)
    plt.show()

    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Enter implementation here
    model = MLPClassifier(alpha=0.00001, hidden_layer_sizes=(30, 10), random_state=1)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    return confusion_matrix(Y_test, y_pred)

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
        
    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)
    
    plotErr(err, 100)
    
    cM = confMatrix(X_test,y_test,w)
    
    sciKit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test_Part1()
