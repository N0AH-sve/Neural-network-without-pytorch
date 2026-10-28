import pytest

from my_torch.encoding.labels import LabelSet

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
