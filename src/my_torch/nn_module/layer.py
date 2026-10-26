import numpy as np

class Layer:

    input : int
    output : int
    def init_w_fct(self, input : int, output : int):
        pass

    def __init__(self, input: int, output : int, active_fct : str):
        self.input = input
        self.output = output
        self.name_active_fct = active_fct
        self.w = self.init_w_fct(self.input, self.output)
        self.b = np.zeros(output)



    def active_fct(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass