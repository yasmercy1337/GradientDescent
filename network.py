import numpy as np
import itertools as it
import functools as ft

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
    
    def gradient(self, inputs: np.ndarray, real: np.ndarray, learning_rate: float = 0.01
                 ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ Computes the gradient and adjusts the weights and biases accordingly """
        d_biases = [np.zeros(self.dimensions[layer_num + 1]) for layer_num in range(len(self.layers))]
        d_weights = [np.zeros((self.dimensions[layer_num + 1], self.dimensions[layer_num])) 
                     for layer_num in range(len(self.layers))]

        expected = self.layers[-1].last_activated
        dA = np.array([2 * (a - y) for (a, y) in zip(expected, real)])
        activations = [inputs] + [self.layers[layer_num].last_activated for layer_num in range(len(self.layers))] + [expected]
        for layer_num in reversed(range(len(self.layers))):
            dB = np.zeros(self.dimensions[layer_num + 1])
            dW = np.zeros((self.dimensions[layer_num + 1], self.dimensions[layer_num]))
            for j in range(self.dimensions[layer_num + 1]): # 2
                dA_dZ = activate_derivative(activations[layer_num + 1][j])
                dB[j] = dA[j] * dA_dZ
                for k in range(self.dimensions[layer_num]): # 3
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


def activate(x: float) -> float:
    """ Calls an activation function (ReLU) """
    return max(0, x)

def activate_derivative(x: float) -> float:
    """ Returns 0 if x < 0, else 1 """
    return float(x > 0)

def cost(expected: np.ndarray, real: np.ndarray) -> float:
    """ Mean squared error between real outputs and generated outputs """
    squared_errors = [(x - y) ** 2 for (x, y) in zip(real, expected)]
    return sum(squared_errors) / len(squared_errors)
