import numpy as np
import pytest

from my_torch.encoding.labels import BASIC, FULL, LabelSet

CLASSES = ["Nothing", "Check White", "Checkmate Black"]


@pytest.fixture
def labels():
    return LabelSet(CLASSES)


def test_classes_keep_their_order(labels):
    assert labels.classes == tuple(CLASSES)
    assert len(labels) == 3


@pytest.mark.parametrize("index, name", list(enumerate(CLASSES)))
def test_index_and_label_round_trip(labels, index, name):
    assert labels.index(name) == index
    assert labels.label(index) == name


@pytest.mark.parametrize("label", ["check white", "CHECK WHITE", " Check   White ", "Check\tWhite"])
def test_lookup_ignores_case_and_spaces(labels, label):
    assert labels.index(label) == 1


def test_accepts_any_iterable():
    assert LabelSet(name for name in CLASSES).classes == tuple(CLASSES)


@pytest.mark.parametrize("label", ["Stalemate", "Checkmate", "Check Whit", ""])
def test_unknown_label_lists_the_valid_ones(labels, label):
    with pytest.raises(ValueError, match="expected one of: Nothing, Check White, Checkmate Black"):
        labels.index(label)


def test_duplicate_classes_are_rejected():
    with pytest.raises(ValueError, match="duplicate class"):
        LabelSet(["Check", "check"])


def test_presets():
    assert BASIC.classes == ("Nothing", "Check", "Checkmate")
    assert FULL.classes == (
        "Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black",
    )
    assert not BASIC.coloured
    assert FULL.coloured


@pytest.mark.parametrize("label, index", [
    ("Nothing", 0),
    ("Check", 1),
    ("Check White", 1),
    ("check  black", 1),
    ("Checkmate", 2),
    ("Checkmate White", 2),
    ("Checkmate Black", 2),
])
def test_basic_ignores_the_colour(label, index):
    assert BASIC.index(label) == index


@pytest.mark.parametrize("label", ["Check", "Checkmate"])
def test_full_requires_the_colour(label):
    with pytest.raises(ValueError, match="unknown label"):
        FULL.index(label)


def test_full_keeps_colours_apart():
    assert FULL.index("Checkmate White") != FULL.index("Checkmate Black")


def test_a_lone_colour_is_not_a_label():
    with pytest.raises(ValueError, match="unknown label"):
        BASIC.index("White")


def test_one_hot_has_a_single_one_per_row():
    y = FULL.one_hot(["Check Black", "Nothing", "Checkmate White"])
    assert y.shape == (3, len(FULL))
    assert np.array_equal(y.sum(axis=1), [1.0, 1.0, 1.0])
    assert np.array_equal(y.argmax(axis=1), [2, 0, 3])


def test_one_hot_uses_the_lookup_rules():
    y = BASIC.one_hot(["checkmate BLACK", " Check  White "])
    assert np.array_equal(y, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def test_one_hot_accepts_a_generator():
    assert BASIC.one_hot(label for label in ["Check", "Nothing"]).shape == (2, 3)


def test_one_hot_of_nothing_is_an_empty_matrix():
    assert BASIC.one_hot([]).shape == (0, 3)


def test_one_hot_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unknown label 'Stalemate'"):
        BASIC.one_hot(["Check", "Stalemate"])
