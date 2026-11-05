import numpy as np
import pytest

from my_torch.analyser.network import load_network
from my_torch.encoding.features import INPUT_SIZE
from my_torch.encoding.labels import BASIC, FULL
from my_torch.nn_module.neural_network_module import Neural_network_module


def saved(tmp_path, sizes):
    path = tmp_path / "net.nn"
    Neural_network_module(sizes).save(str(path))
    return str(path)


@pytest.mark.parametrize("sizes", [
    [INPUT_SIZE, len(BASIC)],
    [INPUT_SIZE, 16, len(BASIC)],
    [INPUT_SIZE, 64, 32, 16, len(BASIC)],
])
def test_a_network_that_fits_is_loaded(tmp_path, sizes):
    network = load_network(saved(tmp_path, sizes))
    assert [layer.input for layer in network.layers] == sizes[:-1]


def test_the_loaded_network_is_the_saved_one(tmp_path):
    original = Neural_network_module([INPUT_SIZE, 8, len(BASIC)])
    path = tmp_path / "net.nn"
    original.save(str(path))
    x = np.zeros((2, INPUT_SIZE))
    assert np.allclose(load_network(str(path)).forward(x), original.forward(x))


def test_a_coloured_network_is_loaded(tmp_path):
    network = load_network(saved(tmp_path, [INPUT_SIZE, 8, len(FULL)]))
    assert network.layers[-1].output == len(FULL)


@pytest.mark.parametrize("inputs", [768, 770, 4])
def test_wrong_input_size(tmp_path, inputs):
    with pytest.raises(ValueError, match=f"takes {inputs} inputs, expected 769"):
        load_network(saved(tmp_path, [inputs, 8, len(BASIC)]))


@pytest.mark.parametrize("outputs", [1, 2, 4, 6])
def test_wrong_output_size(tmp_path, outputs):
    with pytest.raises(ValueError, match=f"{outputs} output neurons, expected one of: 3, 5"):
        load_network(saved(tmp_path, [INPUT_SIZE, 8, outputs]))


def test_error_names_the_file(tmp_path):
    path = saved(tmp_path, [768, 8, 3])
    with pytest.raises(ValueError, match="net.nn"):
        load_network(path)


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        load_network(str(tmp_path / "missing.nn"))
