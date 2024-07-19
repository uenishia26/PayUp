import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1) # set a seed so that the results are consistant

X, Y = load_planar_dataset()


"""
Functions: 
    modelFunction, predictFunction, initalizeFunction, ForwardPropogation, BackwardPropogation, computeCost 

Matrix w/h their respective dimensions: 
    X:  (2,400) (inputs, trainingExamples)
    Y:  (1, 400) (Red or Blue, trainingExamples)
    w1: (hiddenLayer, inputs)
    b1: (hiddenLayer,1)
    z1: (hiddenLayer, trainingExamples)
    a1: (hiddenLayer, trainingExamples)
    w2: (1, hiddenLayer)
    z2: (1, trainingExample)
    a2: (1, trainingExample)
    b2: (1,1)
    h_n: 4
    dw1: (hiddenLayer, input)
    dw2: (1,hiddenLayer)
    db1: (hiddenLayer,1)
    db2: (1,1)
    dz1: (hiddenLayer, trainingExample)
    dz2: (1,trainingExample)
"""
#Sigmoid function
def sigmoid(z): 
   return 1/(1+np.exp(-z))

#Return parameters dictionary includes random weights/biases intialized
def initalize(X, n_h):
    np.random.seed(2)
    w1 = np.random.randn(n_h, X.shape[0]) * 0.01
    w2 = np.random.randn(1,n_h) * 0.01
    b1 = np.zeros((n_h,1))
    b2 = np.zeros((1,1))
    parameters = {"w1":w1, "w2":w2, "b1":b1, "b2":b2}
    return parameters


def forwardPropagation(parameters, X): 
    #np.dot = matrix multiplication (dim determined by outside dimensions)
    #Retriving values from dictionary
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    #Forward Propgation 
    z1 = np.dot(w1,X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = sigmoid(z2)

    intermComp = {"z1":z1, "a1":a1, "z2":z2, "a2":a2} #Intermediate computation
    return intermComp, parameters

def backwardPropagation(parameters,intermComp, x, Y, learningRate):
    m = Y.shape[1]
    w1,b1,w2,b2 = parameters["w1"], parameters["b1"], parameters["w2"], parameters["b2"]
    z1,a1,z2,a2 = intermComp["z1"], intermComp["a1"], intermComp["z2"], intermComp["a2"]

    #First back propogation
    dz2 = a2 - Y #(1,trainingExample)
    dw2 = 1/m * np.dot(dz2,a1.T) #(1,hiddenLayer)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True) 

    #Second back propogation
    dz1 = np.dot(w2.T,dz2) * (1-np.tanh(z1)**2) #(hiddenLayer, trainingExample) * (hiddenLayer, trainingExamples)
    dw1 = 1/m * np.dot(dz1, x.T) #(hiddenLayer, input)
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True) #(hiddenLayer, 1)

    gradient = {"dw2":dw2,"db2":db2,"dw1":dw1,"db1":db1}
    return gradient

#Cost function
def computeCost(A2, y):
    m = A2.shape[1]
    logprobs = np.multiply(np.log(A2),y)+np.multiply((1-y),np.log(1-A2))
    cost = - np.sum(logprobs)/m
    
    cost = float(np.squeeze(cost))
    return cost

#Predict Function takes y^
def predict(A2): 
    prediction = np.zeros((1,A2.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0,i] > 0.5: 
            prediction[0,i] = 1
        else: 
            prediction[0,i] = 0 
    return prediction

#parameters["w1"], parameters["b1"], parameters["w2"], parameters["b2"]
def optimize(X,y, parameters, epoch, learningRate): 
    for numIter in range(epoch): #Training
        intermComp, parameters = forwardPropagation(parameters, X)
        gradient = backwardPropagation(parameters,intermComp, X, y, learningRate)
        w1,b1,w2,b2 = parameters["w1"], parameters["b1"], parameters["w2"], parameters["b2"]
        #reCalculating weights/biases
        dw1 = gradient["dw1"]
        dw2 = gradient["dw2"]
        db1 = gradient["db1"]
        db2 = gradient["db2"]

        w1 = w1 - (learningRate*dw1)
        w2 = w2 - (learningRate*dw2)
        b1 = b1 - (learningRate*db1)
        b2 = b2 - (learningRate*db2) #Recreating parameters dictionary with new values
        parameters = {"w1":w1, "b1":b1, "w2":w2, "b2":b2}

        if numIter % 1000 == 0: 
            print(f"Current Cost {numIter}: {computeCost(intermComp["a2"],y)}")

    return intermComp, parameters


def model(X,y, epoch=1000, learningRate=1.5): 
   parameters = initalize(X,4)  #Initalizing with four hidden layers 
   intermComp,parameters = optimize(X,y,parameters, epoch, learningRate) #Parameters stores the optimal weight/biases

   a2 = intermComp["a2"] 
   prediction = predict(a2) #Get the prediction 

   

model(X,Y, epoch= 11000, learningRate=1.5)

   
       
       
       





