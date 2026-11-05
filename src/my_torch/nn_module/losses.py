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


def log_softmax(output):
    """log(softmax(output)) of each row. The row's largest score is subtracted
    first, which leaves the result unchanged but keeps exp from overflowing."""
    shifted = output - output.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(output):
    """Raw scores turned into probabilities: positive, summing to 1 per row."""
    return np.exp(log_softmax(output))


def cross_entropy_value(output, target):
    """Minus the log of the probability softmax gives to the target class,
    averaged over the samples. output holds the raw scores of a linear last
    layer: softmax is applied here, not by the network."""
    return float(-np.sum(target * log_softmax(output)) / len(output))


def cross_entropy_gradient(output, target):
    return (softmax(output) - target) / len(output)


class Loss:

    def __init__(self, name, value, gradient):
        self.name = name
        self.value = value
        self.gradient = gradient


LOSSES = {
    "mse": Loss("mse", mse_value, mse_gradient),
    "cross_entropy": Loss("cross_entropy", cross_entropy_value, cross_entropy_gradient),
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
