import pytest

from my_torch.nn_module.layer import Layer
from my_torch.nn_module.neural_network_module import Neural_network_module


def test_layer_keeps_its_dimensions():
    layer = Layer(768, 64, "relu")
    assert layer.input == 768
    assert layer.output == 64


def test_layer_has_one_bias_per_output_neuron():
    layer = Layer(5, 3, "relu")
    assert len(layer.b) == 3


def test_layer_biases_start_at_zero():
    """Biases start at zero; only the weights need symmetry breaking."""
    layer = Layer(5, 3, "relu")
    assert all(value == 0.0 for value in layer.b)


def test_layer_remembers_its_activation_name():
    assert Layer(5, 3, "tanh").name_active_fct == "tanh"


@pytest.mark.parametrize("sizes, expected", [
    ([2, 1], 1),
    ([4, 5, 3], 2),
    ([768, 128, 64, 3], 3),
])
def test_network_builds_one_layer_per_connection(sizes, expected):
    """A network over n sizes has n-1 layers: layers sit between the sizes."""
    assert len(Neural_network_module(sizes).layers) == expected


def test_network_wires_consecutive_sizes():
    network = Neural_network_module([768, 128, 64, 3])
    assert [(l.input, l.output) for l in network.layers] == [
        (768, 128), (128, 64), (64, 3),
    ]


def test_network_output_of_a_layer_feeds_the_next():
    """The invariant that makes the forward pass composable."""
    layers = Neural_network_module([768, 128, 64, 3]).layers
    for current, following in zip(layers, layers[1:]):
        assert current.output == following.input
