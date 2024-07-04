import random
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):

        # the quadratic cost equation to calculation the lost of the neural net
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):

        # get the backprop delta error to change weights
        return (a-y) * sigmoid_prime(z)
    
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
            
        # the cross entropy equation to calculation the lost of tghe neural net
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
       
    @staticmethod
    def delta(z, a, y):
            
        # get the backprop delta error to change weights. there no derivative of sigmoid beacause it get canceld out when derived
        return (a-y)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):

        # intiate the netural net with the given layers 
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost= cost

    def default_weight_initializer(self):

        # intiate biases as a random number between 0 and the standard deviation of 1
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # intiate weights as a random number between 0 and the standard deviation of 1 but divided by the sqrt of the number of weights to that neruon
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    def large_weight_intializer(self):

        # intiate biases and weights as a random number between 0 and the standard deviation of 1
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):

        # take the input of the perivous layer (a) and do the matrix multipulaction of weights and a and add the biases
        # plug the value into sigmoid and output to next layer
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)

        return a
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0, evaluation_data=None, monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, monitor_training_cost=False, monitor_training_accuracy=False):
        
        # if evalution_data is given create get the length of the evaluation_data
        if evaluation_data: n_data = len(evaluation_data)
        # initate the n as the number of data in the training_data and all the lsit to monitor cost and accuracy
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for j in range(epochs):
            # Randomly shuffle the training_data
            random.shuffle(training_data)
            # Using the mini_batchs_size parameters create each mini_batches 
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # for each mini_batch we run the update_mini_batches which calulates the updated weights and baises
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print (f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print (f"Accuracy on training data: {accuracy} / {n}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):

        # create a empty matrices to hold the vallue of the weight and baises reductaion calculated by backprop
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # take the pervious value held by nabla matrices and add it to the value calaculated by the backprop
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update the biases and wiegths with values calculated    
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        # the (1-eta*(lmbda/n)) is a L2 regularazation that helps the neural network generalize the data by forcing the wieghts to be small
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        # create a empty matrices to hold the vallue of the weight and baises reductaion calculated by backprop
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # intialize the activation with x which is the value of the output of each neuro
        activation = x
        # intialize the activation list with x which is the value of the output of each neuron 
        activations = [x]
        # intialize the zs list with the z vectors of each layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            # calculate the activation for each node using matrix multiplication
            z = np.dot(w, activation)+b
            # append each list with z vector and activation value
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # take the last z vector, activation value, and the actual y value to calucate the loss delta with the derivative of the chossen los function
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # add the value to th end of the nabla matrix
        nabla_b[-1] = delta
        # beacuse of chain rule you need to do a matrix multiplaction with the pervious activation value
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):

        if convert: results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else: results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def total_cost(self, data, lmbda, convert=False):
        
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost
    

def sigmoid(z):

    # the sigmoid equation to use as activation function
    return 1.0/(1.0+np.exp(-z))
    
def sigmoid_prime(z):

    # the derivative of the sigmoid afunction used to backprop   
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0

    return e

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 10])

mec, mea, mtc, mta = net.SGD(training_data[:1000], 200, 10, 5.0, 0.025, evaluation_data=test_data[:100], monitor_evaluation_cost=True)

plt.plot(mec)
plt.show()