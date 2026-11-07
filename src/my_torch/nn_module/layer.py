import numpy as np
from my_torch.nn_module import activations

class Layer:

    input : int
    output : int
    def init_w_fct(self, input : int, output : int):
        rng = np.random.default_rng()
        return rng.normal(0.0, 0.1, (input, output))

    def __init__(self, input: int, output : int, active_fct : str, w=None, b=None,
                 dropout : float = 0.0):
        self.input = input
        self.output = output
        self.name_active_fct = active_fct
        self.activation = activations.get(active_fct)
        self.w = self.init_w_fct(self.input, self.output) if w is None else np.array(w, dtype=float)
        self.b = np.zeros(output) if b is None else np.array(b, dtype=float)
        self.dropout = dropout
        self.mask = None


    def forward(self, x, rng : np.random.Generator | None = None):
        """Activations of the layer; with an rng, a new dropout mask is drawn
        and applied, which is what makes this a training pass.

        Each unit is dropped on its own for each sample, so every sample of a
        batch goes through a different sub-network. The survivors are divided
        by the keep rate, so the sum reaching the next layer keeps the scale it
        has when nothing is dropped: a pass without an rng then needs no
        correction of its own.
        """
        self.x = x
        self.z = x @ self.w + self.b
        a = self.activation.forward(self.z)
        if rng is None or self.dropout == 0.0:
            self.mask = None
            return a
        keep = 1.0 - self.dropout
        self.mask = rng.random(a.shape) < keep
        return a * self.mask / keep

    def backward(self, grad):
        """Gradient of the loss with respect to the layer's input.

        A unit dropped in the forward pass took no part in the output, so it
        must receive no gradient: the mask of that very pass is applied again,
        with the same scaling.
        """
        if self.mask is not None:
            grad = grad * self.mask / (1.0 - self.dropout)
        delta = self.activation.backward(self.z, grad)
        self.grad_w = self.x.T @ delta
        self.grad_b = delta.sum(axis=0)
        return delta @ self.w.T

    def update(self, pas):
        self.w = self.w - pas * self.grad_w
        self.b = self.b - pas * self.grad_b
