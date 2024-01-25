import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), X_train.T), y_train)

    return w

def mse(X_train, y_train, w):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    y_pred = pred(X_train, w)
    
    return np.mean((y_train - y_pred) ** 2)

def pred(X_i, w):
    pred = np.dot(X_i, w)
    return pred

def test_SciKit(X_train, X_test, Y_train, Y_test):
    linear_regression_model = linear_model.LinearRegression()
    
    linear_regression_model.fit(X_train, Y_train)
    y_pred = linear_regression_model.predict(X_test)
    return mean_squared_error(Y_test, y_pred)

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
