import copy
import math

import numpy as np
import pytest

from my_torch.nn_module.neural_network_module import Neural_network_module

N_SAMPLES = 10


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_SAMPLES, 4))
    # Column 0 holds the sample's id, so a batch can tell which rows it got.
    x[:, 0] = np.arange(N_SAMPLES)
    y = rng.normal(size=(N_SAMPLES, 2))
    return x, y


def network(**hyperparameters):
    return Neural_network_module([4, 3, 2], learning_rate=0.01, **hyperparameters)


def record_batches(net):
    """Wrap net.forward to remember the sample ids of every batch it sees."""
    batches = []
    forward = net.forward

    def recording_forward(x):
        batches.append(x[:, 0].astype(int).tolist())
        return forward(x)

    net.forward = recording_forward
    return batches


def weights(net):
    return [layer.w for layer in net.layers] + [layer.b for layer in net.layers]


def test_a_batch_as_big_as_the_data_is_one_full_batch_step(data):
    """batch_size >= len(x) must still give the old full-batch descent."""
    x, y = data
    net = network(epochs=1, batch_size=N_SAMPLES)
    by_hand = copy.deepcopy(net)
    net.train(x, y)
    by_hand.backward((by_hand.forward(x) - y) / N_SAMPLES)
    by_hand.update(by_hand.learning_rate)
    for trained, expected in zip(weights(net), weights(by_hand), strict=True):
        assert np.allclose(trained, expected)


@pytest.mark.parametrize("batch_size, sizes", [
    (4, [4, 4, 2]),
    (5, [5, 5]),
    (1, [1] * N_SAMPLES),
    (64, [N_SAMPLES]),
])
def test_an_epoch_cuts_the_data_into_batches(data, batch_size, sizes):
    x, y = data
    net = network(epochs=1, batch_size=batch_size)
    batches = record_batches(net)
    net.train(x, y)
    assert [len(batch) for batch in batches] == sizes


def test_one_update_per_batch(data, monkeypatch):
    x, y = data
    net = network(epochs=3, batch_size=4)
    updates = []
    monkeypatch.setattr(net, "update", lambda rate: updates.append(rate))
    net.train(x, y)
    assert len(updates) == 3 * math.ceil(N_SAMPLES / 4)


def test_every_sample_is_used_once_per_epoch(data):
    x, y = data
    net = network(epochs=2, batch_size=4)
    batches = record_batches(net)
    net.train(x, y)
    for epoch in (batches[:3], batches[3:]):
        assert sorted(i for batch in epoch for i in batch) == list(range(N_SAMPLES))


def test_the_order_is_shuffled_and_changes_every_epoch(data):
    x, y = data
    net = network(epochs=2, batch_size=N_SAMPLES)
    batches = record_batches(net)
    net.train(x, y)
    first, second = batches
    assert first != list(range(N_SAMPLES))
    assert first != second


def test_the_same_seed_gives_the_same_weights(data):
    x, y = data
    net = network(epochs=5, batch_size=3, seed=7)
    twin = copy.deepcopy(net)
    net.train(x, y)
    twin.train(x, y)
    for a, b in zip(weights(net), weights(twin), strict=True):
        assert np.array_equal(a, b)


def test_another_seed_gives_other_weights(data):
    x, y = data
    net = network(epochs=5, batch_size=3, seed=7)
    other = copy.deepcopy(net)
    other.seed = 8
    net.train(x, y)
    other.train(x, y)
    assert not np.array_equal(net.layers[0].w, other.layers[0].w)


def test_training_lowers_the_loss():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4))
    y = x @ rng.normal(size=(4, 2))
    net = Neural_network_module([4, 2], epochs=50, learning_rate=0.05, batch_size=16)
    before = np.mean((net.forward(x) - y) ** 2)
    net.train(x, y)
    assert np.mean((net.forward(x) - y) ** 2) < before / 10


@pytest.mark.parametrize("n_samples, batch_size", [
    (10, 4),      # last slice smaller
    (10, 5),      # slices fit exactly
    (10, 4096),   # one slice
    (1, 1),
])
def test_forward_in_batches_equals_forward(n_samples, batch_size):
    x = np.random.default_rng(3).normal(size=(n_samples, 4))
    net = network()
    assert np.allclose(net.forward_in_batches(x, batch_size), net.forward(x))


def test_forward_in_batches_keeps_the_order(data):
    x, _ = data
    net = network()
    batches = record_batches(net)
    net.forward_in_batches(x, batch_size=3)
    assert [i for batch in batches for i in batch] == list(range(N_SAMPLES))
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]


def test_forward_in_batches_of_nothing():
    assert network().forward_in_batches(np.empty((0, 4))).shape == (0, 2)


def test_forward_in_batches_reads_booleans():
    x = np.random.default_rng(4).random((9, 4)) < 0.5
    net = network()
    assert np.allclose(net.forward_in_batches(x, 2), net.forward(x.astype(float)))
