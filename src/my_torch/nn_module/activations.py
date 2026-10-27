import numpy as np


def _as_array(z):
    return np.asarray(z, dtype=np.float64)


def linear_forward(z):
    return _as_array(z)


def linear_backward(z, grad):
    return _as_array(grad)


def sigmoid_forward(z):
    """1 / (1 + exp(-z)), in (0, 1)."""
    z = _as_array(z)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_backward(z, grad):
    a = sigmoid_forward(z)
    return _as_array(grad) * a * (1.0 - a)


def tanh_forward(z):
    """Hyperbolic tangent, in (-1, 1). Zero-centred, unlike the sigmoid."""
    return np.tanh(_as_array(z))


def tanh_backward(z, grad):
    a = np.tanh(_as_array(z))
    return _as_array(grad) * (1.0 - a * a)


def relu_forward(z):
    """max(0, z). Cheap and does not saturate for positive inputs."""
    return np.maximum(_as_array(z), 0.0)


def relu_backward(z, grad):
    # The derivative is undefined at exactly 0; 0 is the usual convention.
    return np.where(_as_array(z) > 0.0, _as_array(grad), 0.0)


class Activation:

    def __init__(self, name, forward, backward):
        self.name = name
        self.forward = forward
        self.backward = backward


ACTIVATIONS = {
    "linear": Activation("linear", linear_forward, linear_backward),
    "sigmoid": Activation("sigmoid", sigmoid_forward, sigmoid_backward),
    "tanh": Activation("tanh", tanh_forward, tanh_backward),
    "relu": Activation("relu", relu_forward, relu_backward),
}

def get(name):
    if isinstance(name, Activation):
        return name
    key = str(name).strip().lower()
    if key not in ACTIVATIONS:
        valid = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"unknown activation {name!r}, expected one of: {valid}")
    return ACTIVATIONS[key]


def names():
    return sorted(ACTIVATIONS)
