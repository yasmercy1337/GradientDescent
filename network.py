import numpy as np
import itertools as it
import functools as ft

Gradient = tuple[list[np.ndarray], list[np.ndarray]]

class Layer:
    """ A matrix of weights and biases """
    def __init__(self, num_inputs: int, num_outputs: int):
        """ Initializes a matrix of weights and biases """
        self.weights = np.random.rand(num_outputs, num_inputs)
        self.biases = np.random.rand(num_outputs)
        self.last_activated = np.random.rand(num_outputs)
    
    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """ Calls an activation function on the outputs of the layer """
        self.last_activated = np.array([activate(x) for x in (self.weights @ inputs + self.biases)])
        return self.last_activated
    
    def __repr__(self) -> str:
        """ Returns a string representation of this layer """
        return f"Weights: {self.weights}\nBiases: {self.biases}"
    
class Network:
    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_layers: int, layer_size: int) -> None:
        """ Initializes an array of layers """
        self.dimensions = [num_inputs] + [layer_size] * num_hidden_layers + [num_outputs]
        self.layers = [Layer(inputs, outputs) for (inputs, outputs) in it.pairwise(self.dimensions)]
        
    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """ Feeds the input through each layer of this network """
        for layer in self.layers:
            inputs = layer.evaluate(inputs)
        return inputs
    
    def __repr__(self) -> str:
        """ Returns a string representation of this network """
        return "\n\n".join(map(repr, self.layers))
    
    def zero_gradient(self) -> Gradient:
        """ Returns a gradient (weights, biases) with only zeros """
        d_biases = [np.zeros(self.dimensions[layer_num + 1]) for layer_num in range(len(self.layers))]
        d_weights = [np.zeros((self.dimensions[layer_num + 1], self.dimensions[layer_num])) 
                     for layer_num in range(len(self.layers))]
        return d_weights, d_biases
    
    def gradient(self, inputs: np.ndarray, real: np.ndarray) -> Gradient:
        """ Computes the gradient and adjusts the weights and biases accordingly """
        d_weights, d_biases = self.zero_gradient()
        expected = self.layers[-1].last_activated
        dA = np.array([2 * (a - y) for (a, y) in zip(expected, real)])
        activations = [inputs] + [self.layers[layer_num].last_activated for layer_num in range(len(self.layers))] + [expected]
        for layer_num in reversed(range(len(self.layers))):
            dB = np.zeros(self.dimensions[layer_num + 1])
            dW = np.zeros((self.dimensions[layer_num + 1], self.dimensions[layer_num]))
            for j in range(self.dimensions[layer_num + 1]):
                dA_dZ = activate_derivative(activations[layer_num + 1][j])
                dB[j] = dA[j] * dA_dZ
                for k in range(self.dimensions[layer_num]):
                    dW[j, k] = dB[j] * activations[layer_num][k]
            
            d_biases[layer_num] += dB
            d_weights[layer_num] += dW

            if layer_num == 0:
                break
            dA_new = np.zeros(self.dimensions[layer_num])
            for j in range(self.dimensions[layer_num + 1]):
                dA_dZ = activate_derivative(activations[layer_num + 1][j])
                for k in range(self.dimensions[layer_num]):
                    dA_new[k] += dA[j] * dA_dZ * self.layers[layer_num].weights[j, k]
            dA = dA_new
        return d_weights, d_biases

    def update(self, gradient: Gradient, learning_rate: float = 0.01):
        for w, b, layer in zip(*gradient, self.layers):
            layer.weights -= w * learning_rate
            layer.biases -= b * learning_rate

    def train(self, inputs: list[np.ndarray], real_outputs: list[np.ndarray], 
              iterations: int = 10, learning_rate: float = 0.000001) -> None:
        """ Takes in labeled dataset and trains for n iterations"""
        for _ in range(iterations):
            dW, dB = self.zero_gradient()
            error = 0
            for (input, real) in zip(inputs, real_outputs):
                error += cost(self.evaluate(input), real) / len(inputs)
                w, b = self.gradient(input, real)
                dW, dB = v_add(dW, w), v_add(dB, b)
            self.update((dW, dB), learning_rate / len(inputs))
            # print(error)
                
def activate(x: float) -> float:
    """ Calls an activation function (ReLU) """
    # return max(0, x)
    # return 1 / (1 + 2.71 ** -x)
    return x

def activate_derivative(x: float) -> float:
    """ Returns 0 if x < 0, else 1 """
    # return float(x > 0)
    # return 1 / ((2.71 ** x) * (1 + 2.71 ** -x) ** 2)
    return 1

def cost(expected: np.ndarray, real: np.ndarray) -> float:
    """ Mean squared error between real outputs and generated outputs """
    squared_errors = [(x - y) ** 2 for (x, y) in zip(real, expected)]
    return sum(squared_errors) / len(squared_errors)

def v_add(x: list, y: list) -> list:
    return [(a + b) for (a, b) in zip(x, y)]