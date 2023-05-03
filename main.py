from network import *
import numpy as np
import random


def create_dataset(n: int) -> tuple[np.ndarray, np.ndarray]:
    """ Creates the inputs and expected outputs"""
    data = list(range(n * 3))
    chosen = random.sample(data, n)

    inputs = np.array([np.array([x]) for x in chosen])
    outputs = np.array(inputs, copy=True)

    return inputs, outputs

def train_test_split(arr: np.ndarray, n: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """ Splits the given dataset into a training and testing """
    cutoff = int(len(arr) * n)
    return arr[:cutoff], arr[cutoff:]

def main():
    net = Network(1, 1, 3, 3)
    # print(net)
    inputs, outputs = create_dataset(10)
    train_in, train_out = train_test_split(inputs)
    test_in, test_out = train_test_split(inputs)

    # for _ in range(10):
    #     out = net.evaluate(np.array([2]))
    #     real = np.array([2])
    #     net.update(net.gradient(np.array([2]), real))

    #     print(out, real, cost(out, real))
        # print(net, "\n")

    net.train(train_in, train_out, iterations=10000)
    # print(net)
    print(net.evaluate(np.array([5])))
    print(net.evaluate(np.array([10])))

if __name__ == "__main__":
    main()