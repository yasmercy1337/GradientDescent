import numpy as np
import itertools as it
import functools as ft

class Layer:
    """ A matrix of weights and biases """
    def __init__(self, num_inputs: int, num_outputs: int):
        """ Initializes a matrix of weights and biases """
        self.weights = np.random.rand(num_outputs, num_inputs)
        self.biases = np.random.rand(num_outputs, 1)
    
    def activate_output(self, output: float) -> float:
        """ Calls an activation function (ReLU) on an output """
        return max(0, output)
    
    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """ Calls an activation function on the outputs of the layer """
        return self.activate_output(self.weights @ inputs + self.biases)
    
    def __repr__(self) -> str:
        """ Returns a string representation of this layer """
        return f"Weights: {self.weights}\nBiases: {self.biases}"
    
class Network:
    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_layers: int, layer_size: int) -> None:
        """ Initializes an array of layers """
        dimensions = [num_inputs] + [layer_size] * num_hidden_layers + [num_outputs]
        self.layers = [Layer(inputs, outputs) for (inputs, outputs) in it.pairwise(dimensions)]
        
    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """ Feeds the input through each layer of this network """
        for layer in self.layers:
            inputs = layer.evaluate(inputs)
        return inputs
    
    def __repr__(self) -> str:
        """ Returns a string representation of this network """
        return "\n\n".join(map(repr, self.layers))
    
    def cost(self, inputs: np.ndarray, real: np.ndarray) -> float:
        """ Mean squared error between real outputs and generated outputs """
        squared_errors = [(x - y) ** 2 for (x, y) in zip(real, self.evaluate(inputs))]
        return sum(squared_errors) / len(squared_errors)
        # NOTE: in practice, this is the mean error for a whole training set, not just a data point
    
    def gradient(self, inputs: np.ndarray, real: np.ndarray) -> np.ndarray:
        """ Computes the gradient for the current neural network (given a data point)"""
        # NOTE: in real gradient descent, this is done for all data points and averaged out
        