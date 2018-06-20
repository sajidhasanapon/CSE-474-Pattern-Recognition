from Layer import Layer
import Functions
import numpy


class Network:

    def __init__(self, size_network):
        """
        :param list size_network:
        """

        self.size_network = size_network
        self.network = []
        self.L = len(size_network) - 1

        self.X = None
        self.Y = None
        self.m = None

        self.index_from = None
        self.activator = None

        for i in range(1, self.L + 1):
            self.network.append(Layer(n_neuron=size_network[i], n_feature=size_network[i-1]))


    def forward(self):
        A = self.X
        for layer in self.network:
            layer.A_prev = A

            Z = numpy.matmul(layer.W_T, A) + layer.b
            A = Functions.activation_function(Z, self.activator)  

            layer.A = A

        return A


    def backward(self, learning_rate):
        """
        :param  double learning_rate:
        :return:
        """
        dA = self.network[-1].A - self.Y

        for i in reversed(range(self.L)):
            layer = self.network[i]

            # dZ = dA * derivative of activation function
            # "*" means element by element multiplication
            dZ = dA * Functions.derivative_activation_function(layer.A, self.activator)
            #dZ = dA * (layer.A * (1 - layer.A))

            dW = (1.0/self.m) * numpy.matmul(dZ, layer.A_prev.T)
            db = (1.0/self.m) * numpy.sum(dZ, axis=1, keepdims=True)

            layer.W_T = layer.W_T - learning_rate * dW  # don't know if it's a call by reference
            layer.b = layer.b - learning_rate * db      # EDIT: yes, it is ! :D 

            dA = numpy.matmul(layer.W_T.T, dZ)


    def train(self, filename, epoch=10000, activator="sigmoid", learning_rate=0.5):

        print("Beginnig training...")

        # samples in data file are in row-vector form
        # we transpose the matrix to convert them to column-vector form
        data = numpy.genfromtxt(filename, delimiter=',').T
        print("Total number of training samples : %d" %(len(data.T)))

        # X = feature matrix; X.shape = (n, m) where n = no. of fatures, m = no. of samples
        self.X = data[[i for i in range(self.size_network[0])], :]

        # Y = ground truth. initially, Y.shape = (1, m)
        # Y.shape needs to converted to (n_classes, m) 
        Y = data[-1, :].tolist()

        # m = number of samples in training set
        self.m = len(Y)

        # for multiple classes, Y has to be a 2D matrix
        # Y[i, j] = 1 if sample j is of class i, else 0
        self.Y = [[0 for i in range(len(Y))] for j in range(self.size_network[-1])]

        # activator function
        # possible options are sigmoid, tanh and, relu
        self.activator = activator

        # the classes are numbered as 
        # <index_from, index_from+1, inndex_from+2 ...> etc.
        # by default, index_from is set to 0 (zero)
        # i.e., classes are numbered as 0, 1, 2, ... etc.
        # in some data sets, classes are indexed from 1
        #i.e., 1, 2, 3, ... etc.
        self.index_from = int(min(Y))
        print("Indexing from %d" %(self.index_from))
        print("===============\n")
        print("Training...")

        for i in range(len(Y)):
            c = int(Y[i])
            self.Y[c - 1*self.index_from][i] = 1

        self.Y = numpy.array(self.Y)

        for i in range(epoch):
            self.forward()
            self.backward(learning_rate)

            Functions.show_progress(i+1, epoch)
        
        print("\n\nTraining Complete")
        print("~~~~~~~~~~~~~~~~~~\n")

    
    def decide(self, X):
        A = numpy.array(X).T
        for layer in self.network:
            layer.A_prev = A

            Z = numpy.matmul(layer.W_T, A) + layer.b
            A = 1.0 / (1.0 + numpy.exp(-Z))  # sigmoid

            layer.A = A

        y_temp = A.T.tolist() 
        y_hat  = []
        for row in y_temp:
            y_hat.append(row.index(max(row)) + 1*self.index_from)

        return y_hat

   
    def test(self, filename):

        print("Beginnig test...")
        print("Testing...", end=" ")
        data = numpy.genfromtxt(filename, delimiter=',').T

        # X = feature matrix; X.shape = (n, m) where n = no. of fatures, m = no. of samples
        X = data[[i for i in range(self.size_network[0])], :].T

        Y = data[-1, :].tolist()
 
        # m = number of samples in training set
        m = len(Y)

        y_hat = self.decide(X)

        n_miss = 0.0
        for i in range(m):
            #print("Expected : %d    Found : %d" %(Y[i], y_hat[i]))
            if (Y[i] != y_hat[i]):
                n_miss += 1

        print("Test complete\n")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Total number of test samples = %d" %(m))
        print("Correctly classified         = %d" %(m -n_miss))
        print("Misclassified                = %d" %(n_miss))

        accuracy = float(m - n_miss) / float(m)
        print("Accuracy                     = %f%%" %(accuracy * 100))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")