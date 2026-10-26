EPOCH = 100
LEARNING_RATE = 0.005
SEED = 42 # everyone have 42 <

# choice Active_fct between sigmoid fcttanh, relU, soft plus expo linera https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/
# choice weight there is a lot of method

class Layer:

    input : int
    output : int
    active_fct : str # have to do a dictionnary to get the fct in function of itd name
    def init_w_fct(input : int, output : int):
        pass

    def __init__(self, input: int, output : int):
        self.input = input
        self.output = output
        self.w = self.init_w_fct(self.input, self.output)
        self.b = 0 # maybe not correct, have to research bit more, maybe do a rnadom, 'll see later
        



    def active_fct():
        pass

    def forward():
        pass

    def backward():
        pass

class Network_neural_module:
    def __init__(self, int_layers : list[int]):
        self.layers = self._init_layers(int_layers)


    def _init_layers(self, int_layers : list[int]) -> list[Layer]:
        layers = []
        for i in range(len(int_layers) - 1):
            next_inputs = int_layers[i + 1]
            layer = Layer(int_layers[i], next_inputs)
            layers.append(layer)
        return layers

    def save(self):
        pass

    def load(self):
        pass

    def train(self):
        pass



        


