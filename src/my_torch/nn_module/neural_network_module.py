from my_torch.nn_module.layer import Layer

EPOCH = 100
LEARNING_RATE = 0.005
SEED = 42 # everyone have 42 <

# choice Active_fct between sigmoid fcttanh, relU, soft plus expo linera https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/
# choice weight there is a lot of method


class Neural_network_module:
    def __init__(self, int_layers : list[int]):
        self.layers = self._init_layers(int_layers)


    def _init_layers(self, int_layers : list[int]) -> list[Layer]:
        layers = []
        for i in range(len(int_layers) - 1):
            next_inputs = int_layers[i + 1]
            activation = "linear" if i == len(int_layers) - 2 else "relu"
            layer = Layer(int_layers[i], next_inputs, activation)
            layers.append(layer)
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, pas):
        for layer in self.layers:
            layer.update(pas)

    def save(self):
        pass

    def load(self):
        pass

    def train(self, x, y, epochs=EPOCH, pas=LEARNING_RATE):
        for _ in range(epochs):
            output = self.forward(x)
            self.backward((output - y) / len(x))
            self.update(pas)
