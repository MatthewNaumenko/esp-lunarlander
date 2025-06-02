import pickle
import os

def save_network(network, filename):
    data = network.get_weights()
    with open(filename, "wb") as f:
        pickle.dump(data, f)