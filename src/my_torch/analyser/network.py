"""Networks loaded by the analyzer, checked against the chess encoding."""

from my_torch.encoding.features import INPUT_SIZE
from my_torch.encoding.labels import for_outputs
from my_torch.nn_module.neural_network_module import Neural_network_module


def load_network(path: str) -> Neural_network_module:
    """Load a .nn file that takes an encoded board and outputs one score per
    class of a label set; the network itself knows nothing about chess."""
    network = Neural_network_module.load(path)
    inputs, outputs = network.layers[0].input, network.layers[-1].output
    if inputs != INPUT_SIZE:
        raise ValueError(f"{path}: network takes {inputs} inputs, expected {INPUT_SIZE}")
    try:
        for_outputs(outputs)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    return network
