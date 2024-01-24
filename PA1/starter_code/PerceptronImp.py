import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):
    max_epochs = 5000

    w = np.zeros(len(X_train[0]) + 1)
    
    ones_to_be_added = np.ones(len(X_train))
    X = np.hstack((np.atleast_2d(ones_to_be_added).T, X_train))
    
    best_w = w
    best_error = float('inf')
    
    for epoch in range(max_epochs):        
        for i, x_i in enumerate(X):
            prediction = pred(x_i, w)
            if prediction != y_train[i]:
                w = w + x_i * y_train[i]
        
        epoch_error = errorPer(X, y_train, w)
        if epoch_error < best_error:
            best_error = epoch_error
            best_w = w
    
    return best_w

def errorPer(X, y_train, w):
    misclassifed = 0
    
    for i, x_i in enumerate(X):
        prediction = pred(x_i, w)
        if prediction != y_train[i]:
            misclassifed += 1
            
    return misclassifed / len(X)        

def pred(X_i, w):
    dot_product = np.dot(X_i, w)
        
    if dot_product > 0:
        return 1
    else:
        return -1
    
def confMatrix(X_train, y_train, w):
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
                matrix[1][0] += 1
            else:
                matrix[1][1] += 1
                
    return matrix
                

def test_SciKit(X_train, X_test, Y_train, Y_test):
    clf = Perceptron(tol=1e-3, max_iter=5000, early_stopping=False)
    
    clf.fit(X_train, Y_train)
    
    y_pred = clf.predict(X_test)
    
    return confusion_matrix(Y_test, y_pred)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
