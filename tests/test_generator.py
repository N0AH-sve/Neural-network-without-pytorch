import json
import re
from pathlib import Path

import pytest

from my_torch.generator.generator import (
    DEFAULT_INIT,
    Generator,
    GeneratorError,
    main,
)
from my_torch.nn_module.neural_network_module import (
    EPOCH,
    LEARNING_RATE,
    SEED,
    Neural_network_module,
)

CONFIG_FILES = Path(__file__).resolve().parent.parent / "config_files"

VALID_CONF = {
    "layers": [4, 3, 2],
    "activations": ["relu", "sigmoid"],
}


def fixture(name: str) -> str:
    return str(CONFIG_FILES / name)


def write_conf(tmp_path, conf, name="net.conf"):
    """Drop a configuration on disk and return its path as a string."""
    path = tmp_path / name
    path.write_text(conf if isinstance(conf, str) else json.dumps(conf))
    return str(path)


# --- parse_args ------------------------------------------------------------

def test_parse_args_pairs_each_config_with_its_count():
    assert Generator.parse_args(["a.conf", "2", "b.conf", "1"]) == [
        ("a.conf", 2),
        ("b.conf", 1),
    ]


def test_parse_args_rejects_an_empty_command_line():
    with pytest.raises(GeneratorError, match="invalid number of arguments"):
        Generator.parse_args([])


def test_parse_args_rejects_a_config_without_its_count():
    with pytest.raises(GeneratorError, match="invalid number of arguments"):
        Generator.parse_args(["a.conf", "2", "b.conf"])


@pytest.mark.parametrize("nb", ["0", "-1", "1.5", "two", "", " 1"])
def test_parse_args_rejects_a_count_that_is_not_a_positive_integer(nb):
    with pytest.raises(GeneratorError, match="invalid number of networks"):
        Generator.parse_args(["a.conf", nb])


# --- get_conf --------------------------------------------------------------

def test_get_conf_reads_a_json_object(tmp_path):
    path = write_conf(tmp_path, VALID_CONF)
    assert Generator([path, "1"]).get_conf(path) == VALID_CONF


def test_get_conf_rejects_a_file_that_is_not_json(tmp_path):
    path = write_conf(tmp_path, "{not json")
    with pytest.raises(GeneratorError, match="invalid JSON"):
        Generator([path, "1"]).get_conf(path)


def test_get_conf_names_the_offending_file(tmp_path):
    path = write_conf(tmp_path, "{not json", name="broken.conf")
    with pytest.raises(GeneratorError, match="broken.conf"):
        Generator([path, "1"]).get_conf(path)


def test_get_conf_lets_a_missing_file_raise_an_os_error(tmp_path):
    missing = str(tmp_path / "nope.conf")
    with pytest.raises(OSError):
        Generator([missing, "1"]).get_conf(missing)


# --- check_conf ------------------------------------------------------------

def check(conf):
    """Validate a configuration the way run() does, without touching the disk."""
    Generator(["net.conf", "1"]).check_conf(conf, "net.conf")


def test_check_conf_accepts_a_minimal_configuration():
    check(VALID_CONF)


def test_check_conf_accepts_the_optional_keys():
    check(VALID_CONF | {
        "init": DEFAULT_INIT,
        "hyperparameters": {"epochs": 10, "learning_rate": 0.5, "seed": 42},
    })


def test_check_conf_rejects_a_configuration_that_is_not_an_object():
    with pytest.raises(GeneratorError, match="must be an object"):
        check([4, 3, 2])


@pytest.mark.parametrize("key", ["layers", "activations"])
def test_check_conf_rejects_a_configuration_missing_a_mandatory_key(key):
    conf = {k: v for k, v in VALID_CONF.items() if k != key}
    with pytest.raises(GeneratorError, match=f"missing key '{key}'"):
        check(conf)


@pytest.mark.parametrize("layers", [[], [4], "4,3", 4])
def test_check_conf_rejects_layers_that_do_not_describe_two_sizes(layers):
    with pytest.raises(GeneratorError, match="at least 2 sizes"):
        check({"layers": layers, "activations": ["relu"]})


@pytest.mark.parametrize("size", ["3", 3.0, None, True])
def test_check_conf_rejects_a_layer_size_that_is_not_an_integer(size):
    with pytest.raises(GeneratorError, match="is not an integer"):
        check({"layers": [4, size], "activations": ["relu"]})


@pytest.mark.parametrize("size", [0, -2])
def test_check_conf_rejects_a_layer_size_that_is_not_positive(size):
    with pytest.raises(GeneratorError, match="must be positive"):
        check({"layers": [4, size], "activations": ["relu"]})


def test_check_conf_rejects_activations_that_are_not_a_list():
    with pytest.raises(GeneratorError, match="must be a list"):
        check({"layers": [4, 3], "activations": "relu"})


@pytest.mark.parametrize("names", [[], ["relu"], ["relu", "relu", "relu"]])
def test_check_conf_wants_one_activation_per_connection(names):
    """3 layer sizes describe 2 connections, so exactly 2 activations."""
    with pytest.raises(GeneratorError, match="one per connection"):
        check({"layers": [4, 3, 2], "activations": names})


def test_check_conf_rejects_an_unknown_activation():
    with pytest.raises(GeneratorError, match="unknown activation"):
        check({"layers": [4, 3], "activations": ["banana"]})


def test_check_conf_reports_which_activation_is_unknown():
    with pytest.raises(GeneratorError, match="activation 1"):
        check({"layers": [4, 3, 2], "activations": ["relu", "banana"]})


def test_check_conf_rejects_hyperparameters_that_are_not_an_object():
    with pytest.raises(GeneratorError, match="'hyperparameters' must be an object"):
        check(VALID_CONF | {"hyperparameters": [10, 0.5]})


@pytest.mark.parametrize("key", ["epochs", "seed", "batch_size", "patience"])
@pytest.mark.parametrize("value", ["10", 10.5, None, True])
def test_check_conf_rejects_a_non_integer_hyperparameter(key, value):
    with pytest.raises(GeneratorError, match=f"'{key}' must be an integer"):
        check(VALID_CONF | {"hyperparameters": {key: value}})


@pytest.mark.parametrize("epochs", [0, -5])
def test_check_conf_rejects_epochs_that_are_not_positive(epochs):
    with pytest.raises(GeneratorError, match="'epochs' must be positive"):
        check(VALID_CONF | {"hyperparameters": {"epochs": epochs}})


@pytest.mark.parametrize("batch_size", [0, -5])
def test_check_conf_rejects_a_batch_size_that_is_not_positive(batch_size):
    with pytest.raises(GeneratorError, match="'batch_size' must be positive"):
        check(VALID_CONF | {"hyperparameters": {"batch_size": batch_size}})


@pytest.mark.parametrize("batch_size", [1, 64, 100_000])
def test_check_conf_accepts_a_positive_batch_size(batch_size):
    """A batch bigger than the dataset is allowed: it means full-batch descent."""
    check(VALID_CONF | {"hyperparameters": {"batch_size": batch_size}})


@pytest.mark.parametrize("patience", [0, -1])
def test_check_conf_rejects_a_patience_that_is_not_positive(patience):
    with pytest.raises(GeneratorError, match="'patience' must be positive"):
        check(VALID_CONF | {"hyperparameters": {"patience": patience}})


@pytest.mark.parametrize("split", ["0.2", None, True, [0.2]])
def test_check_conf_rejects_a_validation_split_that_is_not_a_number(split):
    with pytest.raises(GeneratorError, match="'validation_split' must be a number"):
        check(VALID_CONF | {"hyperparameters": {"validation_split": split}})


@pytest.mark.parametrize("split", [1, 1.5, -0.1])
def test_check_conf_rejects_a_validation_split_out_of_range(split):
    with pytest.raises(GeneratorError, match=re.escape("'validation_split' must be in [0, 1)")):
        check(VALID_CONF | {"hyperparameters": {"validation_split": split}})


@pytest.mark.parametrize("split", [0, 0.0, 0.2, 0.99])
def test_check_conf_accepts_a_validation_split_in_range(split):
    """0 turns validation off; anything below 1 leaves boards to train on."""
    check(VALID_CONF | {"hyperparameters": {"validation_split": split}})


@pytest.mark.parametrize("loss", ["hinge", "", 5, None, True])
def test_check_conf_rejects_an_unknown_loss(loss):
    with pytest.raises(GeneratorError, match="unknown loss") as error:
        check(VALID_CONF | {"hyperparameters": {"loss": loss}})
    assert "cross_entropy" in str(error.value) and "mse" in str(error.value)


@pytest.mark.parametrize("loss", ["mse", "cross_entropy", " MSE "])
def test_check_conf_accepts_a_known_loss(loss):
    check(VALID_CONF | {"hyperparameters": {"loss": loss}})


def test_check_conf_accepts_a_negative_seed():
    """Unlike epochs, a seed has no sign constraint."""
    check(VALID_CONF | {"hyperparameters": {"seed": -42}})


@pytest.mark.parametrize("rate", ["0.5", None, True])
def test_check_conf_rejects_a_learning_rate_that_is_not_a_number(rate):
    with pytest.raises(GeneratorError, match="'learning_rate' must be a number"):
        check(VALID_CONF | {"hyperparameters": {"learning_rate": rate}})


@pytest.mark.parametrize("rate", [0, 0.0, -0.1])
def test_check_conf_rejects_a_learning_rate_that_is_not_positive(rate):
    with pytest.raises(GeneratorError, match="'learning_rate' must be positive"):
        check(VALID_CONF | {"hyperparameters": {"learning_rate": rate}})


def test_check_conf_rejects_an_unknown_init_strategy():
    with pytest.raises(GeneratorError, match="unknown init"):
        check(VALID_CONF | {"init": "xavier"})


# --- run -------------------------------------------------------------------

def test_run_writes_one_file_per_requested_network(tmp_path, monkeypatch):
    path = write_conf(tmp_path, VALID_CONF)
    monkeypatch.chdir(tmp_path)

    Generator([path, "3"]).run()

    assert sorted(p.name for p in tmp_path.glob("*.nn")) == [
        "net_1.nn", "net_2.nn", "net_3.nn",
    ]


def test_run_names_the_networks_after_the_config_file(tmp_path, monkeypatch):
    path = write_conf(tmp_path, VALID_CONF, name="deep.conf")
    monkeypatch.chdir(tmp_path)

    Generator([path, "1"]).run()

    assert (tmp_path / "deep_1.nn").exists()


def test_run_handles_several_configurations(tmp_path, monkeypatch):
    first = write_conf(tmp_path, VALID_CONF, name="a.conf")
    second = write_conf(tmp_path, VALID_CONF, name="b.conf")
    monkeypatch.chdir(tmp_path)

    Generator([first, "2", second, "1"]).run()

    assert sorted(p.name for p in tmp_path.glob("*.nn")) == [
        "a_1.nn", "a_2.nn", "b_1.nn",
    ]


def test_run_writes_nothing_when_a_later_configuration_is_broken(tmp_path,
                                                                 monkeypatch):
    """Every configuration is validated before the first network is written."""
    good = write_conf(tmp_path, VALID_CONF, name="good.conf")
    bad = write_conf(tmp_path, {"layers": [4, 3]}, name="bad.conf")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GeneratorError, match="missing key 'activations'"):
        Generator([good, "1", bad, "1"]).run()

    assert list(tmp_path.glob("*.nn")) == []


def test_run_produces_networks_that_can_be_loaded_back(tmp_path, monkeypatch):
    path = write_conf(tmp_path, VALID_CONF)
    monkeypatch.chdir(tmp_path)

    Generator([path, "1"]).run()
    net = Neural_network_module.load(str(tmp_path / "net_1.nn"))

    assert [layer.input for layer in net.layers] == [4, 3]
    assert [layer.output for layer in net.layers] == [3, 2]
    assert [layer.activation.name for layer in net.layers] == ["relu", "sigmoid"]


def test_run_carries_the_hyperparameters_into_the_network(tmp_path, monkeypatch):
    conf = VALID_CONF | {
        "hyperparameters": {"epochs": 7, "learning_rate": 0.25, "seed": 1234},
    }
    path = write_conf(tmp_path, conf)
    monkeypatch.chdir(tmp_path)

    Generator([path, "1"]).run()
    net = Neural_network_module.load(str(tmp_path / "net_1.nn"))

    assert net.epochs == 7
    assert net.learning_rate == 0.25
    assert net.seed == 1234


def test_run_falls_back_to_the_default_hyperparameters(tmp_path, monkeypatch):
    path = write_conf(tmp_path, VALID_CONF)
    monkeypatch.chdir(tmp_path)

    Generator([path, "1"]).run()
    net = Neural_network_module.load(str(tmp_path / "net_1.nn"))

    assert net.epochs == EPOCH
    assert net.learning_rate == LEARNING_RATE
    assert net.seed == SEED


def test_run_randomizes_each_network_independently(tmp_path, monkeypatch):
    """Two networks from the same config must not share their weights."""
    path = write_conf(tmp_path, VALID_CONF)
    monkeypatch.chdir(tmp_path)

    Generator([path, "2"]).run()
    first = Neural_network_module.load(str(tmp_path / "net_1.nn"))
    second = Neural_network_module.load(str(tmp_path / "net_2.nn"))

    assert not (first.layers[0].w == second.layers[0].w).all()


# --- main ------------------------------------------------------------------

def test_main_returns_zero_on_a_valid_configuration(tmp_path, monkeypatch):
    path = write_conf(tmp_path, VALID_CONF)
    monkeypatch.chdir(tmp_path)

    assert main([path, "1"]) == 0
    assert (tmp_path / "net_1.nn").exists()


def test_main_turns_a_broken_configuration_into_84(tmp_path, monkeypatch, capsys):
    path = write_conf(tmp_path, {"layers": [4, 3]})
    monkeypatch.chdir(tmp_path)

    assert main([path, "1"]) == 84
    assert "missing key 'activations'" in capsys.readouterr().err


def test_main_turns_a_missing_configuration_into_84(tmp_path, capsys):
    assert main([str(tmp_path / "nope.conf"), "1"]) == 84
    assert capsys.readouterr().err.startswith("my_torch_generator:")


def test_main_turns_a_bad_command_line_into_84(capsys):
    assert main(["net.conf"]) == 84
    assert "invalid number of arguments" in capsys.readouterr().err


def test_main_prefixes_every_error_with_the_binary_name(tmp_path, capsys):
    assert main([str(tmp_path / "nope.conf"), "1"]) == 84
    assert capsys.readouterr().err.startswith("my_torch_generator: ")


# --- configuration fixtures ------------------------------------------------

@pytest.mark.parametrize("name", [
    "valid.conf",
    "valid_minimal.conf",
    "valid_partial_hyperparameters.conf",
    "valid_deep.conf",
])
def test_a_valid_fixture_is_accepted(name):
    generator = Generator([fixture(name), "1"])
    generator.check_conf(generator.get_conf(fixture(name)), fixture(name))


@pytest.mark.parametrize("name, message", [
    ("not_an_object.conf", "configuration must be an object"),
    ("no_layers.conf", "missing key 'layers'"),
    ("no_activations.conf", "missing key 'activations'"),
    ("one_layer.conf", "at least 2 sizes"),
    ("layers_not_a_list.conf", "at least 2 sizes"),
    ("layer_size_not_an_integer.conf", "layer size 1 is not an integer"),
    ("layer_size_not_positive.conf", "layer size 1 must be positive"),
    ("activations_not_a_list.conf", "'activations' must be a list"),
    ("too_few_activations.conf", "one per connection"),
    ("unknown_activation.conf", "unknown activation"),
    ("hyperparameters_not_an_object.conf", "'hyperparameters' must be an object"),
    ("epochs_not_an_integer.conf", "'epochs' must be an integer"),
    ("epochs_not_positive.conf", "'epochs' must be positive"),
    ("batch_size_not_an_integer.conf", "'batch_size' must be an integer"),
    ("batch_size_not_positive.conf", "'batch_size' must be positive"),
    ("patience_not_an_integer.conf", "'patience' must be an integer"),
    ("patience_not_positive.conf", "'patience' must be positive"),
    ("validation_split_not_a_number.conf", "'validation_split' must be a number"),
    ("validation_split_out_of_range.conf", "'validation_split' must be in [0, 1)"),
    ("unknown_loss.conf", "unknown loss 'hinge'"),
    ("seed_not_an_integer.conf", "'seed' must be an integer"),
    ("learning_rate_not_a_number.conf", "'learning_rate' must be a number"),
    ("learning_rate_not_positive.conf", "'learning_rate' must be positive"),
    ("unknown_init.conf", "unknown init"),
])
def test_a_broken_fixture_is_rejected(name, message):
    """Every rejection is a GeneratorError carrying a message that names the fault."""
    generator = Generator([fixture(name), "1"])
    with pytest.raises(GeneratorError, match=re.escape(message)):
        generator.check_conf(generator.get_conf(fixture(name)), fixture(name))


def test_the_invalid_json_fixture_is_rejected_by_get_conf():
    """A file that is not JSON never reaches check_conf."""
    generator = Generator([fixture("invalid_json.conf"), "1"])
    with pytest.raises(GeneratorError, match="invalid JSON"):
        generator.get_conf(fixture("invalid_json.conf"))


def test_a_fixture_rejection_names_the_file():
    generator = Generator([fixture("no_layers.conf"), "1"])
    with pytest.raises(GeneratorError, match="no_layers.conf"):
        generator.check_conf(generator.get_conf(fixture("no_layers.conf")),
                             fixture("no_layers.conf"))


# --- the fixtures, end to end through main ---------------------------------

@pytest.mark.parametrize("name", [
    "valid.conf",
    "valid_minimal.conf",
    "valid_partial_hyperparameters.conf",
    "valid_deep.conf",
])
def test_main_generates_a_network_from_a_valid_fixture(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main([fixture(name), "1"]) == 0

    produced = tmp_path / f"{Path(name).stem}_1.nn"
    assert Neural_network_module.load(str(produced)) is not None


@pytest.mark.parametrize("name", [
    "invalid_json.conf",
    "not_an_object.conf",
    "no_layers.conf",
    "no_activations.conf",
    "one_layer.conf",
    "layers_not_a_list.conf",
    "layer_size_not_an_integer.conf",
    "layer_size_not_positive.conf",
    "activations_not_a_list.conf",
    "too_few_activations.conf",
    "unknown_activation.conf",
    "hyperparameters_not_an_object.conf",
    "epochs_not_an_integer.conf",
    "epochs_not_positive.conf",
    "batch_size_not_an_integer.conf",
    "batch_size_not_positive.conf",
    "patience_not_an_integer.conf",
    "patience_not_positive.conf",
    "validation_split_not_a_number.conf",
    "validation_split_out_of_range.conf",
    "unknown_loss.conf",
    "seed_not_an_integer.conf",
    "learning_rate_not_a_number.conf",
    "learning_rate_not_positive.conf",
    "unknown_init.conf",
])
def test_main_rejects_a_broken_fixture_without_writing_anything(name, tmp_path,
                                                                monkeypatch,
                                                                capsys):
    monkeypatch.chdir(tmp_path)

    assert main([fixture(name), "1"]) == 84
    assert capsys.readouterr().err.startswith("my_torch_generator: ")
    assert list(tmp_path.glob("*.nn")) == []


def test_the_valid_fixture_carries_its_hyperparameters_into_the_network(tmp_path,
                                                                       monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main([fixture("valid.conf"), "1"]) == 0
    net = Neural_network_module.load(str(tmp_path / "valid_1.nn"))

    assert (net.epochs, net.learning_rate, net.seed) == (50, 0.01, 7)


def test_the_minimal_fixture_falls_back_to_the_default_hyperparameters(tmp_path,
                                                                      monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main([fixture("valid_minimal.conf"), "1"]) == 0
    net = Neural_network_module.load(str(tmp_path / "valid_minimal_1.nn"))

    assert (net.epochs, net.learning_rate, net.seed) == (EPOCH, LEARNING_RATE, SEED)


def test_a_hyperparameter_in_the_fixture_wins_over_the_constant(tmp_path,
                                                                monkeypatch):
    """valid_partial_hyperparameters.conf sets only the learning rate."""
    monkeypatch.chdir(tmp_path)

    assert main([fixture("valid_partial_hyperparameters.conf"), "1"]) == 0
    net = Neural_network_module.load(
        str(tmp_path / "valid_partial_hyperparameters_1.nn")
    )

    assert net.learning_rate == 0.2
    assert (net.epochs, net.seed) == (EPOCH, SEED)


def test_the_deep_fixture_produces_the_shape_it_describes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main([fixture("valid_deep.conf"), "1"]) == 0
    net = Neural_network_module.load(str(tmp_path / "valid_deep_1.nn"))

    assert [layer.input for layer in net.layers] == [64, 32, 16, 8]
    assert [layer.output for layer in net.layers] == [32, 16, 8, 1]
    assert [layer.activation.name for layer in net.layers] == [
        "relu", "relu", "tanh", "sigmoid",
    ]


def test_a_configured_batch_size_reaches_the_generated_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conf = tmp_path / "batched.conf"
    conf.write_text(json.dumps(VALID_CONF | {"hyperparameters": {"batch_size": 16}}))
    assert main([str(conf), "1"]) == 0
    assert Neural_network_module.load("batched_1.nn").batch_size == 16


def test_configured_validation_settings_reach_the_generated_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conf = tmp_path / "validated.conf"
    conf.write_text(json.dumps(
        VALID_CONF | {"hyperparameters": {"validation_split": 0.1, "patience": 7}}))
    assert main([str(conf), "1"]) == 0
    network = Neural_network_module.load("validated_1.nn")
    assert (network.validation_split, network.patience) == (0.1, 7)


def test_a_configured_loss_reaches_the_generated_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conf = tmp_path / "squared.conf"
    conf.write_text(json.dumps(VALID_CONF | {"hyperparameters": {"loss": "mse"}}))
    assert main([str(conf), "1"]) == 0
    assert Neural_network_module.load("squared_1.nn").loss == "mse"
