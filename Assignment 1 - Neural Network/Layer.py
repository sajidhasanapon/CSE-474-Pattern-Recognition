import numpy


class Layer:

    def __init__(self, n_neuron, n_feature):
        self.W_T = numpy.random.rand(n_neuron, n_feature)
        self.b = numpy.random.rand(n_neuron, 1)

        self.Z = None
        self.A = None

        self.dZ = None
        self.dA = None

        self.A_prev = None
