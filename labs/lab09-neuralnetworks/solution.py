# -*- coding: utf-8 -*-
"""
A solution to the neural net models lab.

@author: Glenn Bruns
"""

import numpy as np

def sigmoid(x):
    """ Return the value of the sigmoid function on input x. """
    
    return 1/(1 + np.exp(-x))

def run_neuron(x, neuron):
    """ Return the output of the neuron on input x, assumed to be augmented.
    neuron is a NumPy array of the same size as x. 
    """
    
    return sigmoid(x.dot(neuron))

def run_layer(x, layer):
    """ Return the output of the layer on input x, assumed to be augmented.
    layer is a list of numpy arrays, one for each neuron in the layer.
    """
    
    return np.array([run_neuron(x, neuron) for neuron in layer ])

def run_network(x, network):
    """ Return the output of the network on input x. 
    network is a list of list of numpy arrays, one for each layer.
    """
    
    for layer in network:
        xa = np.concatenate([[1], x])
        x = run_layer(xa, layer)
        
    return x

def make_network(input_size, layers):
    """ Randomly initialize a new feedforward neural network.
    Layers is a list containing the number of neurons in each layers. """
    
    w = []
    for layer_size in layers:
        w.append([np.random.rand(input_size + 1) for _ in range(layer_size)])
        input_size = layer_size
        
    return w

# Create and initialize a network with a hidden layer of size 3,
# followed by the output layer, which is a single neuron with
# sigmoid activation.
net = make_network(input_size=3, layers=[3, 1])

# run the network on input [1,2,3]
x = np.array([1,2,3])
run_network(x, net)

def run_network_small(x, network):
    """ A compact version of run_network, just as an illustration. """
    
    for layer in network:
        xa = np.concatenate([[1], x])
        x = np.array([sigmoid(xa.dot(neuron)) for neuron in layer ])
        
    return x    

# test run_network_small
run_network_small(x, net)
