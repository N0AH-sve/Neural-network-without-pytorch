"""Networks loaded by the analyzer, checked against the chess encoding."""

from my_torch.encoding.features import INPUT_SIZE
from my_torch.encoding.labels import BASIC
from my_torch.nn_module.neural_network_module import Neural_network_module


def load_network(path: str) -> Neural_network_module:
    """Load a .nn file that takes an encoded board and outputs one score per
    BASIC class; the network itself knows nothing about chess."""
    network = Neural_network_module.load(path)
    inputs, outputs = network.layers[0].input, network.layers[-1].output
    if inputs != INPUT_SIZE:
        raise ValueError(f"{path}: network takes {inputs} inputs, expected {INPUT_SIZE}")
    if outputs != len(BASIC):
        classes = ", ".join(BASIC.classes)
        raise ValueError(f"{path}: network has {outputs} outputs, expected "
                         f"{len(BASIC)} ({classes})")
    return network
