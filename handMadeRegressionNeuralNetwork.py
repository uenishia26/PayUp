import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#Code for loading dataset is taken from kaggle MUHAMMED DALKIRAN
def load_dataset():
    train_dataset = h5py.File('/Users/anamuuenishi/Downloads/Anamu/muData/catvnoncat/train_catvnoncat.h5', "r")
    test_dataset = h5py.File('/Users/anamuuenishi/Downloads/Anamu/muData/catvnoncat/test_catvnoncat.h5', "r")
    
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])
    classes = np.array(train_dataset["list_classes"][:])
    
    train_dataset.close()
    test_dataset.close()
    
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset() 

#Shape of train_set_x_orig = [209, 64, 64, 3]
#train_set_y = [209,]
#train_set_x = [209,12288]
#test_set_x = [50, 12288]
#Data trained on 209 inputs, tested on 50 



train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = (train_set_x_flatten / 255.)
test_set_x = (test_set_x_flatten / 255.)
train_set_y = train_set_y.reshape((1,-1))
test_set_y = test_set_y.reshape((1,-1))


""" Dimension of each Matrix
X = [numInputFeatures, numSamples]   
w = [numInputFeatures,1]
A = [1, numSample]
y = [209,]

"""

#Match the weights array 
def initalizeMatrix(X): 
    w = np.zeros((X.shape[0],1)) #Two parenthese used for 2d Matrix using zeros 
    b = 0.0
    return w, b

#Activation Function 
def sigmoid(z): 
    return 1/(1+np.exp(-z))

#Forward and backward propogation (Used for training model)
def propogate(w,b,X,y): 
    m = X.shape[1]
    
    #Forward Propogation 
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    #Backward Propogation (Calculating gradient in terms of w and b)
    dw = 1/m * np.dot(X,(A-y).T)
    db = 1/m * (np.sum(A-y))

    gradient = {"dw":dw, 
                "db":db}
    
    return A, gradient
    
#Adjust weights and biases (Used for training model)
def optimize(X, y, w,b, epoch, learning_rate): 
    #Getting the current cost 
    m = X.shape[1]

    for i in range(epoch): 
        A, gradient = propogate(w,b,X,y)
        cost = -1/m * np.sum(y* np.log(A) + (1-y) * np.log(1-A)) 
        #Getting weights from dictionary 
        dw = gradient["dw"]
        db = gradient["db"]

        #Adjusting weights 
        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)

        print(f"Current cost: {cost}")
    
    return w, b 

def predict(X,w,b): 
    #Calculating y hat 
    A = sigmoid(np.dot(w.T, X) + b)
    #Initalizing Prediction Array 
    prediction = np.zeros((X.shape[1],1))

    for i in range(X.shape[1]):
        # >0.5 is cat / <0.5 is not Cat
        if A[0,i] > 0.5: 
            prediction[i,0] = 1
        else: 
            prediction[i,0] = 0 
    
    return prediction 


def model(x_train, y_train, learningRate, epoch): 
    #Initalizing matrix to all Zeros 
    w,b = initalizeMatrix(x_train)

    #Optimized weights and biases 
    w,b = optimize(x_train, y_train,w,b, epoch, learningRate)

    return w,b

#Plotting depending on 
f1Score = []
for x in range(1000, 11000, 1000): 
    w,b = model(train_set_x, train_set_y, 0.1, x)
    ypredict = predict(test_set_x,w,b) #Shape is (DataInput,1)
    ypredict = ypredict.reshape((-1))
    test_set_y = test_set_y.reshape((-1))
    f1Score.append(f1_score(ypredict, test_set_y))

print(f1Score)
chartX = np.arange(1000,11000,1000)
plt.figure(figsize=(10,6))
plt.plot(chartX,f1Score)
plt.xlabel("Epoch inc 1000-10000")
plt.ylabel("f1Score")
plt.show()




    
    






