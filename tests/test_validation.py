import copy
import dataclasses

import numpy as np
import pytest

from my_torch.nn_module.neural_network_module import EpochMetrics, Neural_network_module

N_SAMPLES = 10


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_SAMPLES, 4))
    # Column 0 holds the sample's id, so a forward pass can tell which rows it got.
    x[:, 0] = np.arange(N_SAMPLES)
    y = np.eye(2)[rng.integers(0, 2, N_SAMPLES)]
    return x, y


def network(**hyperparameters):
    return Neural_network_module([4, 3, 2], learning_rate=0.01, **hyperparameters)


def record_forward(net):
    """Wrap net.forward to remember the sample ids of every call, in order."""
    calls = []
    forward = net.forward

    def recording_forward(x):
        calls.append(x[:, 0].astype(int).tolist())
        return forward(x)

    net.forward = recording_forward
    return calls


def test_validation_samples_are_never_trained_on(data):
    """Each epoch: two training batches of 4, then one validation pass of 2."""
    x, y = data
    net = network(epochs=3, batch_size=4, validation_split=0.2)
    calls = record_forward(net)
    net.train(x, y)
    epochs = [calls[i:i + 3] for i in range(0, len(calls), 3)]
    assert len(epochs) == 3
    validation = epochs[0][2]
    assert len(validation) == 2
    for first, second, checked in epochs:
        assert sorted(first + second) == sorted(set(range(N_SAMPLES)) - set(validation))
        assert checked == validation


@pytest.mark.parametrize("n_samples, split, held_out", [
    (10, 0.2, 2),
    (10, 0.5, 5),
    (3, 0.2, 1),     # round(0.6)
    (1, 0.9, 0),     # at least one sample is always left to train on
    (10, 0.0, 0),
])
def test_how_many_samples_are_held_out(n_samples, split, held_out):
    x = np.zeros((n_samples, 4))
    x[:, 0] = np.arange(n_samples)
    net = network(epochs=1, batch_size=n_samples, validation_split=split)
    calls = record_forward(net)
    net.train(x, np.zeros((n_samples, 2)))
    assert len(calls[0]) == n_samples - held_out
    assert sum(len(call) for call in calls[1:]) == held_out


def test_one_metrics_per_epoch(data):
    x, y = data
    history = network(epochs=4).train(x, y)
    assert [m.epoch for m in history] == [1, 2, 3, 4]
    for m in history:
        assert m.loss > 0
        assert m.validation_loss > 0
        assert 0 <= m.validation_accuracy <= 1


def test_without_validation_the_fields_are_none(data):
    x, y = data
    history = network(epochs=2, validation_split=0.0).train(x, y)
    assert all(m.validation_loss is None and m.validation_accuracy is None
               for m in history)


def test_validation_metrics_describe_the_network_after_the_epoch(data):
    x, y = data
    net = network(epochs=2, batch_size=4)
    calls = record_forward(net)
    last = net.train(x, y)[-1]
    held_out = calls[-1]
    output = net.forward(x[held_out])
    assert last.validation_loss == pytest.approx(np.mean((output - y[held_out]) ** 2))
    expected = np.mean(output.argmax(axis=1) == y[held_out].argmax(axis=1))
    assert last.validation_accuracy == pytest.approx(expected)


def test_training_loss_goes_down():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4))
    y = x @ rng.normal(size=(4, 2))
    history = Neural_network_module([4, 2], epochs=30, learning_rate=0.05,
                                    batch_size=16).train(x, y)
    assert history[-1].loss < history[0].loss / 10
    assert history[-1].validation_loss < history[0].validation_loss / 10


def test_the_same_seed_gives_the_same_history(data):
    x, y = data
    net = network(epochs=3, batch_size=3, seed=5)
    twin = copy.deepcopy(net)
    assert net.train(x, y) == twin.train(x, y)


def test_another_seed_holds_out_other_samples(data):
    x, y = data
    first, second = network(epochs=1, seed=1), network(epochs=1, seed=2)
    first_calls, second_calls = record_forward(first), record_forward(second)
    first.train(x, y)
    second.train(x, y)
    assert sorted(first_calls[-1]) != sorted(second_calls[-1])


def test_epoch_metrics_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        EpochMetrics(1, 0.5).loss = 0.1


def script(net, validation_losses):
    """Replace the epoch work with a script: epoch e sets every weight to e and
    reports the e-th validation loss. Lets a test choose exactly when the
    validation loss improves, and see which epoch's weights end up kept."""
    epochs = iter(range(1, len(validation_losses) + 1))
    losses = iter(validation_losses)

    def train_epoch(x, y, order):
        epoch = next(epochs)
        for layer in net.layers:
            layer.w = np.full_like(layer.w, epoch)
        return 1.0

    net._train_epoch = train_epoch
    net._validate = lambda x, y: (next(losses), 0.5)


def kept_epoch(net):
    return int(net.layers[0].w[0, 0])


@pytest.mark.parametrize("losses, patience, stopped_at, kept", [
    ([5, 4, 3, 3.5, 3.2, 3.1, 1], 3, 6, 3),       # 3 epochs without beating 3
    ([5, 4, 4.5, 4.4, 3, 3.1, 3.2, 3.3], 3, 8, 5),  # epoch 5 resets the count
    ([2, 2, 2, 2], 3, 4, 1),                       # equal is not better
    ([5, 6, 1], 1, 2, 1),                          # patience 1: first miss stops
])
def test_training_stops_after_patience_epochs_without_improvement(data, losses, patience,
                                                                  stopped_at, kept):
    x, y = data
    net = network(epochs=len(losses), patience=patience)
    script(net, losses)
    history = net.train(x, y)
    assert history[-1].epoch == stopped_at
    assert kept_epoch(net) == kept


def test_improving_every_epoch_runs_them_all(data):
    x, y = data
    net = network(epochs=4, patience=1)
    script(net, [5, 4, 3, 2])
    assert len(net.train(x, y)) == 4
    assert kept_epoch(net) == 4


def test_the_best_weights_are_kept_even_without_stopping(data):
    """The epoch limit came first, but the last epoch was not the best one."""
    x, y = data
    net = network(epochs=3, patience=5)
    script(net, [3, 2, 2.5])
    assert len(net.train(x, y)) == 3
    assert kept_epoch(net) == 2


def test_without_validation_every_epoch_runs(data):
    x, y = data
    assert len(network(epochs=6, validation_split=0.0, patience=1).train(x, y)) == 6


def test_on_random_labels_training_stops_and_keeps_the_best_epoch():
    """Nothing general can be learned from random labels: the validation loss
    soon stops improving, so training must stop long before the limit."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 4))
    x[:, 0] = np.arange(60)
    y = np.eye(2)[rng.integers(0, 2, 60)]
    net = Neural_network_module([4, 32, 2], epochs=500, learning_rate=0.05,
                                batch_size=8, patience=3)
    calls = record_forward(net)
    history = net.train(x, y)
    best = min(history, key=lambda m: m.validation_loss)
    assert history[-1].epoch < 500
    assert history[-1].epoch - best.epoch == 3
    held_out = calls[-1]
    output = net.forward(x[held_out])
    assert np.mean((output - y[held_out]) ** 2) == pytest.approx(best.validation_loss)


def test_on_epoch_receives_every_epoch_in_order(data):
    x, y = data
    received = []
    history = network(epochs=4).train(x, y, on_epoch=received.append)
    assert received == history


def test_on_epoch_is_called_as_each_epoch_ends(data):
    """Progress is live: epoch n is reported before epoch n + 1 starts."""
    x, y = data
    net = network(epochs=3, validation_split=0.0)
    events = []
    train_epoch = net._train_epoch

    def recording_train_epoch(*args):
        events.append("train")
        return train_epoch(*args)

    net._train_epoch = recording_train_epoch
    net.train(x, y, on_epoch=lambda metrics: events.append(metrics.epoch))
    assert events == ["train", 1, "train", 2, "train", 3]


def test_on_epoch_also_sees_the_epoch_that_stops_training(data):
    x, y = data
    net = network(epochs=10, patience=2)
    script(net, [3, 2, 2.5, 2.4, 1, 1, 1, 1, 1, 1])
    received = []
    net.train(x, y, on_epoch=received.append)
    assert [m.epoch for m in received] == [1, 2, 3, 4]


@pytest.mark.parametrize("losses, patience, kept", [
    ([5, 4, 3, 3.5, 3.2, 3.1], 3, 3),
    ([3, 2, 2.5], 5, 2),
    ([2, 2, 2, 2], 3, 1),
])
def test_history_best_is_the_epoch_whose_weights_were_kept(data, losses, patience, kept):
    x, y = data
    net = network(epochs=len(losses), patience=patience)
    script(net, losses)
    history = net.train(x, y)
    assert history.best.epoch == kept_epoch(net) == kept
    assert history.best is history[kept - 1]


def test_without_validation_there_is_no_best_epoch(data):
    x, y = data
    assert network(epochs=2, validation_split=0.0).train(x, y).best is None
