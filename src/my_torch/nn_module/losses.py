"""Loss functions: how far the outputs of a network are from the targets.

Each loss gives its value, averaged over the samples, and its gradient with
respect to the outputs, which already includes the 1 / N of that average: it
is what the last layer's backward pass receives.
"""

import numpy as np


def mse_value(output, target):
    """Half the squared error of a sample, summed over its outputs, averaged
    over the samples: the half makes the gradient exactly output - target."""
    return float(0.5 * np.sum((output - target) ** 2) / len(output))


def mse_gradient(output, target):
    return (output - target) / len(output)


class Loss:

    def __init__(self, name, value, gradient):
        self.name = name
        self.value = value
        self.gradient = gradient


LOSSES = {
    "mse": Loss("mse", mse_value, mse_gradient),
}


def get(name):
    if isinstance(name, Loss):
        return name
    key = str(name).strip().lower()
    if key not in LOSSES:
        valid = ", ".join(sorted(LOSSES))
        raise ValueError(f"unknown loss {name!r}, expected one of: {valid}")
    return LOSSES[key]


def names():
    return sorted(LOSSES)
