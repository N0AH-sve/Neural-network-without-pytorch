import numpy as np
import pytest

from my_torch.nn_module import losses


@pytest.fixture
def sample():
    """Raw outputs of 4 samples over 3 classes, and one-hot targets."""
    rng = np.random.default_rng(0)
    output = rng.normal(0.0, 2.0, size=(4, 3))
    target = np.eye(3)[[0, 2, 1, 2]]
    return output, target


@pytest.mark.parametrize("name", losses.names())
def test_gradient_matches_numerical_gradient(name, sample):
    """gradient must equal d value / d output, by centred difference."""
    output, target = sample
    loss = losses.get(name)
    h = 1e-6
    numeric = np.empty_like(output)
    for index in np.ndindex(output.shape):
        plus, minus = output.copy(), output.copy()
        plus[index] += h
        minus[index] -= h
        numeric[index] = (loss.value(plus, target) - loss.value(minus, target)) / (2 * h)
    assert np.allclose(loss.gradient(output, target), numeric, atol=1e-6)


@pytest.mark.parametrize("name", losses.names())
def test_gradient_has_the_shape_of_the_output(name, sample):
    output, target = sample
    assert losses.get(name).gradient(output, target).shape == output.shape


@pytest.mark.parametrize("name", losses.names())
def test_value_is_a_float(name, sample):
    assert isinstance(losses.get(name).value(*sample), float)


def test_mse_of_a_perfect_output_is_zero():
    target = np.eye(3)
    assert losses.get("mse").value(target, target) == 0.0
    assert np.array_equal(losses.get("mse").gradient(target, target), np.zeros((3, 3)))


def test_mse_is_half_the_summed_squared_error_per_sample():
    output = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    # sample 1: 0.5 * (1 + 1) = 1, sample 2: 0, average 0.5
    assert losses.get("mse").value(output, target) == 0.5


def test_mse_gradient_is_the_error_over_the_number_of_samples():
    output, target = np.array([[3.0, 0.0]]), np.array([[1.0, 0.0]])
    assert np.array_equal(losses.get("mse").gradient(output, target), [[2.0, 0.0]])


def test_lookup_is_case_and_space_insensitive():
    assert losses.get(" MSE ") is losses.get("mse")


def test_lookup_accepts_a_loss_instance():
    mse = losses.get("mse")
    assert losses.get(mse) is mse


def test_unknown_loss_lists_the_valid_names():
    with pytest.raises(ValueError, match="unknown loss 'mae', expected one of: ") as error:
        losses.get("mae")
    assert all(name in str(error.value) for name in losses.names())


def test_softmax_gives_probabilities(sample):
    output, _ = sample
    probabilities = losses.softmax(output)
    assert np.all(probabilities > 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_softmax_keeps_the_order_of_the_scores(sample):
    """So argmax of the raw scores is already the predicted class."""
    output, _ = sample
    assert np.array_equal(losses.softmax(output).argmax(axis=1), output.argmax(axis=1))


def test_softmax_ignores_a_shift_of_all_scores(sample):
    output, _ = sample
    assert np.allclose(losses.softmax(output), losses.softmax(output + 100.0))


@pytest.mark.parametrize("scores, target_class, expected", [
    ([0.0, 0.0, 0.0], 0, np.log(3)),            # no preference: 1/3 each
    ([np.log(2), 0.0, 0.0], 0, np.log(2)),      # probability 2/4 = 1/2
    ([20.0, 0.0, 0.0], 0, 0.0),                 # sure and right
    ([20.0, 0.0, 0.0], 1, 20.0),                # sure and wrong
])
def test_cross_entropy_is_minus_the_log_probability_of_the_target(scores, target_class,
                                                                 expected):
    value = losses.get("cross_entropy").value(np.array([scores]), np.eye(3)[[target_class]])
    assert value == pytest.approx(expected, abs=1e-6)


def test_cross_entropy_averages_over_the_samples():
    output = np.zeros((2, 3))
    target = np.eye(3)[[0, 1]]
    assert losses.get("cross_entropy").value(output, target) == pytest.approx(np.log(3))


def test_cross_entropy_gradient_is_softmax_minus_target(sample):
    output, target = sample
    expected = (losses.softmax(output) - target) / len(output)
    assert np.allclose(losses.get("cross_entropy").gradient(output, target), expected)


def test_cross_entropy_gradient_rows_sum_to_zero(sample):
    """Raising one class's probability lowers the others by as much."""
    output, target = sample
    assert np.allclose(losses.get("cross_entropy").gradient(output, target).sum(axis=1), 0.0)


def test_huge_scores_do_not_overflow():
    scores = np.array([[1000.0, -1000.0, 0.0]])
    target = np.eye(3)[[1]]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = losses.get("cross_entropy").value(scores, target)
        gradient = losses.get("cross_entropy").gradient(scores, target)
    assert value == pytest.approx(2000.0)
    assert np.allclose(gradient, [[1.0, -1.0, 0.0]])
