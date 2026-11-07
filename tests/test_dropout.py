"""Dropout: a random part of each layer's units is left out of every training
pass, and nothing is left out of any other pass."""

import json

import numpy as np
import pytest

from my_torch.nn_module.layer import Layer
from my_torch.nn_module.neural_network_module import DROPOUT, Neural_network_module


def data(n: int = 40, inputs: int = 6, classes: int = 3):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, inputs))
    y = np.eye(classes)[rng.integers(0, classes, n)]
    return x, y


# --- the layer -------------------------------------------------------------

def test_a_pass_without_an_rng_drops_nothing():
    """Prediction must not be random: no rng, no mask."""
    layer = Layer(4, 100, "linear", dropout=0.5)
    layer.w = np.full((4, 100), 0.25)
    out = layer.forward(np.ones((1, 4)))
    assert layer.mask is None
    assert np.array_equal(out, np.ones((1, 100)))


def test_a_pass_with_an_rng_drops_about_the_given_share():
    layer = Layer(4, 20000, "linear", dropout=0.5)
    layer.forward(np.ones((1, 4)), np.random.default_rng(1))
    kept = layer.mask.mean()
    assert 0.48 < kept < 0.52


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9])
def test_dropping_keeps_the_scale_of_the_activations(rate):
    """The point of dividing by the keep rate: on average the next layer
    receives sums of the same size whether units were dropped or not. Without
    it, training and prediction would feed it scales a keep rate apart."""
    layer = Layer(50, 4000, "linear", dropout=rate)
    layer.w = np.full((50, 4000), 0.01)
    x = np.ones((200, 50))
    whole = layer.forward(x).sum(axis=1).mean()
    dropped = layer.forward(x, np.random.default_rng(2)).sum(axis=1).mean()
    assert dropped == pytest.approx(whole, rel=0.02)


def test_without_the_scaling_the_two_passes_would_disagree():
    """What the division buys, stated the other way round: the raw masked sum
    is a keep rate short of the sum the prediction pass computes."""
    layer = Layer(50, 2000, "linear", dropout=0.5)
    layer.w = np.full((50, 2000), 0.01)
    x = np.ones((200, 50))
    whole = layer.forward(x).sum(axis=1).mean()
    layer.forward(x, np.random.default_rng(7))
    raw = (layer.activation.forward(layer.z) * layer.mask).sum(axis=1).mean()
    assert raw == pytest.approx(0.5 * whole, rel=0.02)


def test_a_zero_rate_never_draws_a_mask():
    """A network without dropout must behave exactly as before, down to the
    draws it does not make from the rng."""
    layer = Layer(4, 8, "relu", dropout=0.0)
    rng = np.random.default_rng(3)
    layer.forward(np.ones((2, 4)), rng)
    assert layer.mask is None
    assert rng.random() == np.random.default_rng(3).random()


def test_every_sample_of_a_batch_gets_its_own_mask():
    """Dropout is drawn per sample, so a batch trains many sub-networks."""
    layer = Layer(4, 200, "relu", dropout=0.5)
    layer.forward(np.ones((2, 4)), np.random.default_rng(4))
    assert not np.array_equal(layer.mask[0], layer.mask[1])


def test_a_dropped_unit_receives_no_gradient():
    """A unit that took no part in the output must take none of the blame."""
    layer = Layer(5, 30, "linear", dropout=0.5)
    layer.forward(np.ones((4, 5)), np.random.default_rng(5))
    layer.backward(np.ones((4, 30)))
    reached = layer.x.T @ (np.ones((4, 30)) * layer.mask / 0.5)
    assert np.allclose(layer.grad_w, reached)


def test_the_mask_is_cleared_by_a_pass_without_an_rng():
    """A stale mask must not leak into the next backward pass."""
    layer = Layer(4, 8, "relu", dropout=0.5)
    layer.forward(np.ones((2, 4)), np.random.default_rng(6))
    layer.forward(np.ones((2, 4)))
    assert layer.mask is None


# --- the network -----------------------------------------------------------

def test_dropout_defaults_to_off():
    assert DROPOUT == 0.0
    assert Neural_network_module([4, 3, 2]).dropout == 0.0


def test_the_output_layer_never_drops():
    """Dropping a score would only add noise to the prediction itself."""
    network = Neural_network_module([4, 8, 6, 2], dropout=0.5)
    assert [layer.dropout for layer in network.layers] == [0.5, 0.5, 0.0]


def test_set_dropout_changes_the_rate_of_a_built_network():
    """What a benchmark needs to compare rates from the same initial weights."""
    network = Neural_network_module([4, 8, 6, 2])
    network.set_dropout(0.5)
    assert network.dropout == 0.5
    assert [layer.dropout for layer in network.layers] == [0.5, 0.5, 0.0]


def test_set_dropout_back_to_zero_turns_it_off():
    network = Neural_network_module([4, 8, 2], dropout=0.5)
    network.set_dropout(0.0)
    layer = network.layers[0]
    layer.forward(np.ones((2, 4)), np.random.default_rng(8))
    assert layer.mask is None


def test_a_single_layer_network_drops_nothing():
    assert Neural_network_module([4, 2], dropout=0.5).layers[0].dropout == 0.0


def test_validation_sees_the_whole_network(monkeypatch):
    """_validate runs during training, so it must still turn dropout off."""
    network = Neural_network_module([6, 16, 3], dropout=0.5, epochs=1,
                                    validation_split=0.5)
    masks = []
    validate = network._validate

    def recording(x, y):
        metrics = validate(x, y)
        masks.append([layer.mask for layer in network.layers])
        return metrics

    monkeypatch.setattr(network, "_validate", recording)
    network.train(*data())
    assert masks and all(mask is None for row in masks for mask in row)


def test_training_with_dropout_still_lowers_the_loss():
    x, y = data(n=200)
    network = Neural_network_module([6, 32, 3], dropout=0.5, epochs=40,
                                    learning_rate=0.05, validation_split=0.0)
    history = network.train(x, y)
    assert history[-1].loss < history[0].loss


def test_dropout_makes_training_differ_from_no_dropout():
    x, y = data(n=100)
    weights = []
    for rate in (0.0, 0.5):
        network = Neural_network_module([6, 16, 3], dropout=rate, epochs=5,
                                        learning_rate=0.05, validation_split=0.0)
        network.layers[0].w = np.full((6, 16), 0.1)
        network.layers[1].w = np.full((16, 3), 0.1)
        network.train(x, y)
        weights.append(network.layers[0].w.copy())
    assert not np.array_equal(*weights)


def test_training_with_dropout_is_reproducible():
    """The masks come from the seeded rng, so the same run gives the same net."""
    x, y = data(n=60)
    runs = []
    for _ in range(2):
        network = Neural_network_module([6, 16, 3], dropout=0.5, epochs=5,
                                        learning_rate=0.05, validation_split=0.0)
        network.layers[0].w = np.full((6, 16), 0.1)
        network.layers[1].w = np.full((16, 3), 0.1)
        network.train(x, y)
        runs.append(network.layers[0].w.copy())
    assert np.array_equal(*runs)


def test_prediction_is_not_random_after_training_with_dropout():
    x, y = data(n=60)
    network = Neural_network_module([6, 16, 3], dropout=0.5, epochs=5,
                                    validation_split=0.0)
    network.train(x, y)
    assert np.array_equal(network.forward(x), network.forward(x))


# --- the file format -------------------------------------------------------

def test_dropout_survives_a_save_and_load(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([4, 3, 2], dropout=0.4).save(str(path))
    assert json.loads(path.read_text())["hyperparameters"]["dropout"] == 0.4
    assert Neural_network_module.load(str(path)).dropout == 0.4


def test_a_reloaded_network_drops_on_the_same_layers(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([4, 8, 6, 2], dropout=0.3).save(str(path))
    loaded = Neural_network_module.load(str(path))
    assert [layer.dropout for layer in loaded.layers] == [0.3, 0.3, 0.0]


@pytest.mark.parametrize("name", [
    "valid.nn",
    "valid_no_hyperparameters.nn",
    "valid_partial_hyperparameters.nn",
])
def test_files_written_before_dropout_get_the_default(name):
    """Every .nn file saved before dropout existed still loads, without it."""
    from pathlib import Path
    fixture = Path(__file__).resolve().parent.parent / "loading_files" / name
    assert Neural_network_module.load(str(fixture)).dropout == DROPOUT


def test_a_config_can_set_the_dropout():
    conf = {"layers": [4, 8, 3], "activations": ["relu", "linear"],
            "hyperparameters": {"dropout": 0.25}}
    assert Neural_network_module.from_config(conf).dropout == 0.25
