"""The game states a network predicts, mapped to its output neurons.

A colour names the side giving check or mate, not the side under attack: in
the subject, Fool's mate (White to move and mated) is "Checkmate Black".
"""

from collections.abc import Iterable

import numpy as np

_COLOURS = ("white", "black")

# The side giving check or mate is the one that has just played, so the side
# to move of a board names the colour of its label by elimination.
_GIVING_CHECK = {"w": "black", "b": "white"}


class LabelSet:
    """Class names in the order of the output neurons."""

    def __init__(self, classes: Iterable[str]):
        self.classes = tuple(classes)
        self._indices = {_normalise(name): i for i, name in enumerate(self.classes)}
        if len(self._indices) != len(self.classes):
            raise ValueError(f"duplicate class in {self.classes}")
        self.coloured = any(_has_colour(name) for name in self._indices)

    def __len__(self) -> int:
        return len(self.classes)

    def index(self, label: str, side: str | None = None) -> int:
        """Output neuron of a label, ignoring case and extra spaces.

        A set without colours drops the colour a label carries; a set with
        colours completes a label given without one from side, the side to
        move of its board. Both ways, the same chess file trains a 3-output
        and a 5-output network.
        """
        key = _normalise(label)
        if not self.coloured and _has_colour(key):
            key = key.rsplit(" ", 1)[0]
        elif self.coloured and key not in self._indices and side in _GIVING_CHECK:
            key = f"{key} {_GIVING_CHECK[side]}"
        try:
            return self._indices[key]
        except KeyError:
            valid = ", ".join(self.classes)
            raise ValueError(f"unknown label {label!r}, expected one of: {valid}") from None

    def label(self, index: int) -> str:
        return self.classes[index]

    def one_hot(self, labels: Iterable[str],
                sides: Iterable[str] | None = None) -> np.ndarray:
        """(N, len(self)) matrix with a single 1 per row, at the label's index.

        sides, the side to move of each board in the same order, lets a
        coloured set complete the labels given without a colour.
        """
        if sides is None:
            indices = [self.index(label) for label in labels]
        else:
            indices = [self.index(label, side)
                       for label, side in zip(labels, sides, strict=True)]
        return np.eye(len(self))[indices]


def _normalise(label: str) -> str:
    return " ".join(label.split()).lower()


def _has_colour(label: str) -> bool:
    """Whether a normalised label ends with a colour."""
    return label.rsplit(" ", 1)[-1] in _COLOURS


BASIC = LabelSet(["Nothing", "Check", "Checkmate"])
FULL = LabelSet(["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"])
LABEL_SETS = {len(BASIC): BASIC, len(FULL): FULL}


def for_outputs(outputs: int) -> LabelSet:
    """Label set of a network with this many output neurons: the .nn file
    stores no class names, so its width is the only hint."""
    try:
        return LABEL_SETS[outputs]
    except KeyError:
        valid = ", ".join(str(size) for size in sorted(LABEL_SETS))
        raise ValueError(f"{outputs} output neurons, expected one of: {valid}") from None
