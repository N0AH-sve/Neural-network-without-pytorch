
W = 0.01
B = 0.01
NB_LAYERS = 1
LAYERS_SIZE = 1

# choice Active_fct between sigmoid fcttanh, relU, soft plus expo linera https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/

class Layer:
    size: 1
    W
    B


    def active_fct():
        pass

    def forward():
        pass

    def backward():
        pass

class Network:
    layers : list[Layer]

class NN_module:
    def create(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def train(self):
        pass