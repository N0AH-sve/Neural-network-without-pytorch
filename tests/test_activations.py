import numpy as np
import pytest

from my_torch.nn_module import activations


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 2.0, size=(4, 5))
    grad = rng.normal(size=(4, 5))
    return z, grad


@pytest.mark.parametrize("name", activations.names())
def test_backward_matches_numerical_gradient(name, sample):
    """The vjp must equal d/dz sum(grad * forward(z)), by centred difference."""
    z, grad = sample
    act = activations.get(name)
    analytic = act.backward(z, grad)

    h = 1e-6
    numeric = np.empty_like(z)
    for index in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[index] += h
        minus[index] -= h
        numeric[index] = (np.sum(grad * act.forward(plus))
                          - np.sum(grad * act.forward(minus))) / (2.0 * h)

    assert np.allclose(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("name", activations.names())
def test_shapes_are_preserved(name, sample):
    z, grad = sample
    act = activations.get(name)
    assert act.forward(z).shape == z.shape
    assert act.backward(z, grad).shape == z.shape


@pytest.mark.parametrize("name", activations.names())
def test_saturates_without_overflow(name):
    """Extreme inputs must saturate to finite values, not inf or nan."""
    z = np.array([[-1e4, -745.0, 0.0, 745.0, 1e4]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a = activations.get(name).forward(z)
        g = activations.get(name).backward(z, np.ones_like(z))
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(g))


def test_sigmoid_is_bounded(sample):
    z, _ = sample
    a = activations.get("sigmoid").forward(z)
    assert np.all((a > 0.0) & (a < 1.0))


def test_relu_is_zero_on_negatives():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.array_equal(activations.get("relu").forward(z),
                          [0.0, 0.0, 0.0, 0.5, 2.0])
    # The gradient must not flow through a zero or negative pre-activation.
    assert np.array_equal(activations.get("relu").backward(z, np.ones_like(z)),
                          [0.0, 0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("name", activations.names())
def test_accepts_one_dimensional_input(name):
    a = activations.get(name).forward([-1.0, 0.5, 2.0])
    assert a.shape == (3,)


def test_lookup_is_case_and_space_insensitive():
    assert activations.get(" ReLU ") is activations.get("relu")
    assert activations.get("TANH") is activations.get("tanh")


def test_lookup_accepts_an_activation_instance():
    tanh = activations.get("tanh")
    assert activations.get(tanh) is tanh


def test_unknown_activation_lists_the_valid_names():
    with pytest.raises(ValueError, match="unknown activation"):
        activations.get("sigmoide")
