import pickle
import os

def save_network(network, filename):
    data = network.get_weights()
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_network(filename):
    import network
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return network.FeedforwardNetwork.from_weights(data)
