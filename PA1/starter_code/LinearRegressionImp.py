import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Function to fit linear regression using the normal equation
def fit_LinRegr(X_train, y_train):
    # Add a column of ones to X_train for the bias term
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # Compute the weights using the normal equation
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), X_train.T), y_train)
    return w

# Mean Squared Error (MSE) calculation
def mse(X_train, y_train, w):
    # Add a column of ones to X_train for the bias term
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # Predict values using the trained weights
    y_pred = pred(X_train, w)
    # Calculate the mean squared error
    return np.mean((y_train - y_pred) ** 2)

# Function to make predictions using weights
def pred(X_i, w):
    pred = np.dot(X_i, w)
    return pred

# Function to test linear regression using scikit-learn
def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Create a linear regression model
    linear_regression_model = linear_model.LinearRegression()
    # Fit the model on the training data
    linear_regression_model.fit(X_train, Y_train)
    # Predict on the test data
    y_pred = linear_regression_model.predict(X_test)
    # Calculate and return the mean squared error
    return mean_squared_error(Y_test, y_pred)

# Subtest function to check robustness against singular matrix
def subtestFn():
    # This function tests if your solution is robust against a singular matrix
    
    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
        w = fit_LinRegr(X_train, y_train)
        print("Weights: ", w)
        print("NO ERROR")
    except:
        print("ERROR")

# Main test function for Part 2
def testFn_Part2():
    # Load the diabetes dataset
    X_train, y_train = load_diabetes(return_X_y=True)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    
    # Train linear regression and get weights
    w = fit_LinRegr(X_train, y_train)
    
    # Testing Part 2a - Calculate MSE
    e = mse(X_test, y_test, w)
    
    # Testing Part 2b - Use scikit-learn to calculate MSE
    scikit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

# Run the subtest function
print('------------------subtestFn----------------------')
subtestFn()

# Run the main test function for Part 2
print('------------------testFn_Part2-------------------')
testFn_Part2()