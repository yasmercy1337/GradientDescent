from network import *
import numpy as np
import random


def create_dataset(n: int) -> tuple[np.ndarray, np.ndarray]:
    """ Creates the inputs and expected outputs"""
    data = list(range(n * 3))
    inputs = np.array(random.sample(data, n))
    outputs = np.array(inputs, copy=True)

    return inputs, outputs

def train_test_split(arr: np.ndarray, n: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """ Splits the given dataset into a training and testing """
    cutoff = int(len(arr) * n)
    return arr[:cutoff], arr[cutoff:]

def main():
    net = Network(1, 1, 3, 3)

    inputs, outputs = create_dataset(10 ** 6)

    for _ in range(10):
        out = net.evaluate(np.array([2]))
        real = np.array([2])
        dW, dB = net.gradient(inputs, real)
        for w, b, layer in zip(dW, dB, net.layers):
            layer.weights -= w * 0.01
            layer.biases -= b * 0.01


        print(out, real, cost(out, real))
        # print(net, "\n")

if __name__ == "__main__":
    main()