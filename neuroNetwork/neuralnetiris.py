import numpy as np
import pandas as pd
import random
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def sigmoid(z):

    # the sigmoid equation to use as activation function
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
def sigmoid_prime(z):

    # the derivative of the sigmoid afunction used to backprop   
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    # the relu equation to use as activation function
    return np.maximum(0, z)

def relu_prime(z):
    # the derivative of the relu function
    return np.where(z > 0, 1, 0)

def vectorized_result(layer_size, j):

    # makes all value expect j to zero
    e = np.zeros((layer_size, 1))
    e[j] = 1.0

    return e

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):

        # the quadratic cost equation to calculation the lost of the neural net
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):

        # The derived version of the quadratic cost equation for backprop
        return (a-y) * sigmoid_prime(z)
    
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
            
        # the cross entropy equation to calculation the lost of tghe neural net
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
       
    @staticmethod
    def delta(z, a, y):
            
        # get the backprop delta error to change weights. there no derivative of sigmoid 
        # beacause it get canceld out when derived
        return (a-y)


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost) -> None:

        random.seed(1)
        # initialize the number of layers of the neural network
        # initialize the weights and biases matrices
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = cost
    
    def feedforward(self, a):

        # Update neuron by taking the input vale and multiplying the weights and add the baises
        # return output to the next neuron
        # we use the dot product so we can move values between layers with different number of neurons
        for b, w in zip(self.biases, self.weights):
            a = relu(np.dot(w, a)+b)
        
        return a
    
    def SGD(self, training_data, epochs, mini_batches_size, lrate, lmbda=0.0, lschd=None, evaluation_data=None, monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, monitor_training_cost=False, monitor_training_accuracy=False):

        # initialize the number of data by the length of the evaluation data
        if evaluation_data: n_data = len(evaluation_data)
        # initialize the numer of data by the length of training data
        n = len(training_data)
        # initialize the list for cost and accuracy to keep track of neural network progress
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = [], [], [], []

        # inital the learning rate variables
        count = 0
        if lschd:
            epochs = sys.maxsize
            traget_lrate = lrate / 128


        for j in range(epochs):

            # if the last 10 epochs had the same cost or accuracy decrease learning rate
            if count == lschd:
                if lrate > traget_lrate: 
                    lrate /= 2
                    count = 0
                else: break
            
            # randomly shuffle the training data
            random.shuffle(training_data)
            # create each minibatches
            mini_batches = [training_data[k:k+mini_batches_size] for k in range(0, n, mini_batches_size)]

            # run update_mini_batch on each minibatch to update weights and biases 
            # before running any minibatch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lrate, lmbda, len(training_data))

            print(f"Epochs {j} training complete")

            # Code to keep track accuracy and cost after each Epoch
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy/n_data)
                print(f"Accuracy on evaluation data: {accuracy/n_data}")
                if j > 0 and evaluation_accuracy[j] <= evaluation_accuracy[j-1]:
                    count += 1
                else:
                    count = 0 

            if monitor_training_accuracy: 
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy/n)
                print(f"Accuracy on training data: {accuracy/n}")
                if j > 0 and training_accuracy[j] <= training_accuracy[j-1]:
                    count += 1
                else:
                    count = 0 

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print(f"Cost on evalution data: {cost}")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            
            print(f"Learning rate: {lrate}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def update_mini_batch(self, mini_batch, lrate, lmbda, n):

        # nabla_b stores the value used to update the biases initate with a matrix of 0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # nabla_w stores the value used to update the weights initate with a matrix of 0
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for each input (x) and output(y) in the minibatch use backprop to calculate the delta loss
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # update the weights and biases based on the calculated dealt loss
        # (1-lrate*(lmbda/n)) is the L2 regularization which keeps the wiegth small so it can 
        # better generalize the data
        # (lrate/len(mini_batch)) is the learning rate which is divided minibatch size to 
        # correlate with the size of minibatch
        self.weights = [(1-lrate*(lmbda/n))*w-(lrate/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lrate/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        # nabla_b stores the value used to update the biases initate with a matrix of 0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # nabla_w stores the value used to update the weights initate with a matrix of 0
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # store the predicted value into a list of activations
        activation = x
        activations = [x]
        # create an empty list of z vectors
        zs = []

        for b, w in zip(self.biases, self.weights):
            # calculate the z vector based on the activation for each biases and weights 
            # add to z vectors list
            z = np.dot(w,activation)+b
            zs.append(z)
            # calculate the activation using the activation for each biases and weights
            activation = relu(z)
            activations.append(activation)
        
        # calculation the delta loss for the output layer using the derivative of the cost function
        # using the last z vector and activation value
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            # get the last z vector 
            z = zs[-l]
            # plug the z vector in the derivative of the activation function 
            sp = relu_prime(z)
            # Using chain rule calculate the delta loss for each layer 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # update the matrices
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False):

        # Run feedforward and get the number of correct predictions
        if convert: results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else: results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        
        # calculate the cost for the entire dataset and normalized based on the data size
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(self.sizes[-1], y)
            cost += (self.cost).fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost
    
    def class_report(self, data, convert=False):

        if convert:
            predictions = [self.feedforward(x) for (x,y) in data]
            predicted_labels = [np.argmax(pred) for pred in predictions]
            true_labels = [np.argmax(y) for (x,y) in data]
        else:
            predictions = [self.feedforward(x) for (x,y) in data]
            predicted_labels = [np.argmax(pred) for pred in predictions]
            true_labels = [y for (x,y) in data]

        return classification_report(true_labels, predicted_labels, target_names=one_hot_encoder.categories_[0])

    
df = pd.read_csv("data/archive/iris_extended.csv")
df = df[['species', 'petal_length', 'petal_width', 'sepal_length', 'sepal_width']]

one_hot_encoder = OneHotEncoder(sparse_output=False)
species_encoded = one_hot_encoder.fit_transform(df[['species']])

df = df.drop(['species'], axis=1)
df = pd.concat([df, pd.DataFrame(species_encoded, columns=one_hot_encoder.get_feature_names_out(['species']))], axis=1)

x = StandardScaler().fit_transform(df[df.columns[:-3]].values) 
y = df[df.columns[-3:]].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123456, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123456, stratify=y_train) # 0.25 x 0.8 = 0.2

y_train = [np.reshape(y, (3, 1)) for y in y_train]
y_test = [np.reshape(y, (3, 1)) for y in y_test]
y_val = [np.reshape(y, (3, 1)) for y in y_val]


X_train = [np.reshape(x, (4, 1)) for x in X_train]
X_test = [np.reshape(x, (4, 1)) for x in X_test]
X_val = [np.reshape(x, (4, 1)) for x in X_val]

train_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))
val_data = list(zip(X_val, y_val))



net = Network([4, 3, 3])
ec, ea, tc, ta = net.SGD(train_data, 30, 100, 0.25, lmbda=10, lschd=5, evaluation_data=val_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)
print(net.class_report(test_data, convert=True))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(ec, label='Training Cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ea, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()