import pandas as pd 
import numpy as np 

""" Dimension of each Matrix
X = [numSample, numInputFeatures]   
w = [1, numInputFeatures]
A = [1, numSample]

"""
def initalizeMatrix(X): 
    w = (np.zeros(X.shape[1]),1)
    b = 0.0
    return w, b

#Activation Function 
def sigmoid(z): 
    return 1/(np.exp(-z))

#Forward and backward propogation (Used for training model)
def propogate(w,b,X,y): 
    m = X.shape[0]

    #Forward Propogation 
    z = np.dot(w.T, X) + b 
    A = sigmoid(z)

    #Backward Propogation (Calculating gradient in terms of w and b)
    dw = 1/m * np.dot(X.T*(A-y))
    db = 1/m * (np.sum(A-y))

    gradient = {"dw":dw, 
                "db":db}
    
    return A, gradient
    
#Adjust weights and biases (Used for training model)
def optimize(X, y, w,b, gradient, epoch, learning_rate): 
    #Getting the current cost 
    m = X.shape[0]

    for i in range(epoch): 
        A, gradient = propogate(w,b,X,y)
        cost = -1/m * np.sum(np.dot(y, np.log(A)) + (1-A) * np.log(1-A)) 
        #Getting weights from dictionary 
        dw = gradient["dw"]
        db = gradient["db"]

        #Adjusting weights 
        w = w - (learning_rate*w)
        b = b - (learning_rate*b)

        print(f"Current cost: {cost}")
    
    return A

def predict(X,y, w,b): 
    #Calculating y hat 
    A = sigmoid(np.dot(w.T, X) + b)
    #Initalizing Prediction Array 
    prediction = (np.zeros([X.shape[0]]),1)
    for i in range(X.shape[0]):
        # >0.5 is cat / <0.5 is not Cat
        if A[0,i] > 0.5: 
            prediction[i,1] = 1
        else: 
            prediction[i,1] = 0 
    
    return prediction 


def model(x_train, x_test, y_train, y_test, X,y, learningRate, epoch): 
    w,b = initalizeMatrix(X.shape[1])
    optimize(x_train, y_train, w, b, )
    






