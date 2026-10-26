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
            layer = Layer(int_layers[i], next_inputs, "ex_act_fct")
            layers.append(layer)
        return layers

    def save(self):
        pass

    def load(self):
        pass

    def train(self):
        pass



        


