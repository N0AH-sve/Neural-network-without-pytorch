import json
from pathlib import Path

import numpy as np
import pytest

from my_torch.nn_module.layer import Layer
from my_torch.nn_module.neural_network_module import (
    BATCH_SIZE,
    EPOCH,
    PATIENCE,
    VALIDATION_SPLIT,
    FORMAT_VERSION,
    LEARNING_RATE,
    SEED,
    Neural_network_module,
)

LOADING_FILES = Path(__file__).resolve().parent.parent / "loading_files"


def fixture(name: str) -> str:
    return str(LOADING_FILES / name)


def test_layer_uses_the_weights_it_is_given():
    layer = Layer(2, 2, "relu", w=[[1.0, 2.0], [3.0, 4.0]], b=[0.5, -0.5])
    assert np.array_equal(layer.w, [[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(layer.b, [0.5, -0.5])


def test_layer_converts_given_weights_to_arrays():
    """The JSON file hands over nested lists; the forward pass needs ndarrays."""
    layer = Layer(2, 2, "relu", w=[[1.0, 2.0], [3.0, 4.0]], b=[0.0, 0.0])
    assert isinstance(layer.w, np.ndarray)
    assert isinstance(layer.b, np.ndarray)


def test_layer_still_randomizes_when_no_weights_are_given():
    layer = Layer(4, 3, "relu")
    assert layer.w.shape == (4, 3)
    assert not np.array_equal(layer.w, np.zeros((4, 3)))


def test_network_hyperparameters_default_to_the_constants():
    network = Neural_network_module([2, 1])
    assert (network.epochs, network.learning_rate, network.seed) == (
        EPOCH, LEARNING_RATE, SEED,
    )


def test_saved_file_is_split_between_data_and_hyperparameters(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([4, 3, 2]).save(str(path))
    saved = json.loads(path.read_text())
    assert saved["version"] == FORMAT_VERSION
    assert set(saved["hyperparameters"]) == {
        "epochs", "learning_rate", "seed", "batch_size", "validation_split", "patience",
        "loss",
    }
    assert len(saved["data"]["layers"]) == 2


def test_saved_layer_carries_sizes_activation_weights_and_biases(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([4, 3, 2]).save(str(path))
    layer = json.loads(path.read_text())["data"]["layers"][0]
    assert set(layer) == {"input", "output", "activation", "w", "b"}
    assert (layer["input"], layer["output"]) == (4, 3)
    assert layer["activation"] == "relu"


def test_loading_a_saved_network_restores_the_weights(tmp_path):
    path = tmp_path / "net.nn"
    network = Neural_network_module([4, 3, 2])
    network.save(str(path))
    loaded = Neural_network_module.load(str(path))
    for original, restored in zip(network.layers, loaded.layers):
        assert np.array_equal(original.w, restored.w)
        assert np.array_equal(original.b, restored.b)
        assert original.name_active_fct == restored.name_active_fct


def test_loading_a_saved_network_restores_the_hyperparameters(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([2, 1], epochs=7, learning_rate=0.25, seed=3).save(str(path))
    loaded = Neural_network_module.load(str(path))
    assert (loaded.epochs, loaded.learning_rate, loaded.seed) == (7, 0.25, 3)


def test_a_reloaded_network_predicts_exactly_like_the_original(tmp_path):
    """What the round trip is really for: identical predictions, not just equal bytes."""
    path = tmp_path / "net.nn"
    network = Neural_network_module([4, 3, 2])
    network.save(str(path))
    x = np.random.default_rng(0).normal(size=(5, 4))
    assert np.array_equal(network.forward(x), Neural_network_module.load(str(path)).forward(x))


def test_training_after_a_reload_keeps_working(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([4, 3, 2], epochs=3, learning_rate=0.01).save(str(path))
    loaded = Neural_network_module.load(str(path))
    x = np.random.default_rng(1).normal(size=(6, 4))
    y = np.random.default_rng(2).normal(size=(6, 2))
    before = loaded.layers[0].w.copy()
    loaded.train(x, y)
    assert not np.array_equal(before, loaded.layers[0].w)


def test_loading_a_fixture_reads_its_layer_topology():
    network = Neural_network_module.load(fixture("valid.nn"))
    assert [(l.input, l.output) for l in network.layers] == [(4, 3), (3, 2)]
    assert [l.name_active_fct for l in network.layers] == ["relu", "linear"]


def test_loading_a_fixture_reads_its_hyperparameters():
    network = Neural_network_module.load(fixture("valid.nn"))
    assert (network.epochs, network.learning_rate, network.seed) == (50, 0.01, 7)


@pytest.mark.parametrize("name", [
    "valid_no_hyperparameters.nn",
    "valid_partial_hyperparameters.nn",
])
def test_missing_hyperparameters_fall_back_to_the_constants(name):
    """Hyperparameters come from the file; absent ones use the module defaults."""
    network = Neural_network_module.load(fixture(name))
    assert network.epochs == EPOCH
    assert network.seed == SEED


def test_a_hyperparameter_present_in_the_file_wins_over_the_constant():
    assert Neural_network_module.load(
        fixture("valid_partial_hyperparameters.nn")
    ).learning_rate == 0.2


@pytest.mark.parametrize("name, message", [
    ("bad_version.nn", "unsupported format version"),
    ("no_data.nn", "no layer found"),
    ("no_layers.nn", "no layer found"),
    ("missing_key_b.nn", "missing key 'b'"),
    ("bad_w_shape.nn", "w has shape"),
    ("bad_b_shape.nn", "b has shape"),
    ("bad_chaining.nn", "layer 0 outputs 3 but layer 1 expects 9"),
    ("unknown_activation.nn", "unknown activation"),
    ("invalid_json.nn", "Expecting value"),
])
def test_a_broken_file_is_rejected(name, message):
    """Every rejection is a ValueError: JSONDecodeError is one too."""
    with pytest.raises(ValueError, match=message):
        Neural_network_module.load(fixture(name))


def test_loading_a_file_that_does_not_exist_raises_an_os_error(tmp_path):
    with pytest.raises(OSError):
        Neural_network_module.load(str(tmp_path / "absent.nn"))


def test_batch_size_defaults_to_the_constant():
    assert Neural_network_module([2, 1]).batch_size == BATCH_SIZE == 64


def test_batch_size_survives_a_save_and_load(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([2, 1], batch_size=16).save(str(path))
    assert json.loads(path.read_text())["hyperparameters"]["batch_size"] == 16
    assert Neural_network_module.load(str(path)).batch_size == 16


@pytest.mark.parametrize("name", [
    "valid.nn",
    "valid_no_hyperparameters.nn",
    "valid_partial_hyperparameters.nn",
])
def test_files_written_before_batch_size_get_the_default(name):
    """Every .nn file saved before batch_size existed still loads."""
    assert Neural_network_module.load(fixture(name)).batch_size == BATCH_SIZE


def test_a_config_can_set_the_batch_size():
    conf = {"layers": [4, 3], "activations": ["relu"], "hyperparameters": {"batch_size": 8}}
    assert Neural_network_module.from_config(conf).batch_size == 8


def test_a_config_without_batch_size_gets_the_default():
    conf = {"layers": [4, 3], "activations": ["relu"]}
    assert Neural_network_module.from_config(conf).batch_size == BATCH_SIZE


def test_validation_defaults_to_the_constants():
    network = Neural_network_module([2, 1])
    assert (network.validation_split, network.patience) == (VALIDATION_SPLIT, PATIENCE)
    assert (VALIDATION_SPLIT, PATIENCE) == (0.2, 3)


def test_validation_settings_survive_a_save_and_load(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([2, 1], validation_split=0.1, patience=5).save(str(path))
    saved = json.loads(path.read_text())["hyperparameters"]
    assert (saved["validation_split"], saved["patience"]) == (0.1, 5)
    loaded = Neural_network_module.load(str(path))
    assert (loaded.validation_split, loaded.patience) == (0.1, 5)


@pytest.mark.parametrize("name", [
    "valid.nn",
    "valid_no_hyperparameters.nn",
    "valid_partial_hyperparameters.nn",
])
def test_files_written_before_validation_get_the_defaults(name):
    network = Neural_network_module.load(fixture(name))
    assert (network.validation_split, network.patience) == (VALIDATION_SPLIT, PATIENCE)


def test_a_config_can_set_the_validation():
    conf = {"layers": [4, 3], "activations": ["relu"],
            "hyperparameters": {"validation_split": 0.0, "patience": 1}}
    network = Neural_network_module.from_config(conf)
    assert (network.validation_split, network.patience) == (0.0, 1)
