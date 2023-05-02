from network import Network
import numpy as np

def main():
    net = Network(1, 1, 0, 1)
    print(net)
    
    out = net.evaluate(np.array([2]))
    print(net.cost(np.array([2]), np.array([2])))

if __name__ == "__main__":
    main()