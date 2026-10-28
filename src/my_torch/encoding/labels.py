"""The game states a network predicts, mapped to its output neurons.

A colour names the side giving check or mate, not the side under attack: in
the subject, Fool's mate (White to move and mated) is "Checkmate Black".
"""

from collections.abc import Iterable

_COLOURS = ("white", "black")


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

    def index(self, label: str) -> int:
        """Output neuron of a label, ignoring case, extra spaces and, when the
        set has no colours, the colour."""
        key = _normalise(label)
        if not self.coloured and _has_colour(key):
            key = key.rsplit(" ", 1)[0]
        try:
            return self._indices[key]
        except KeyError:
            valid = ", ".join(self.classes)
            raise ValueError(f"unknown label {label!r}, expected one of: {valid}") from None

    def label(self, index: int) -> str:
        return self.classes[index]


def _normalise(label: str) -> str:
    return " ".join(label.split()).lower()


def _has_colour(label: str) -> bool:
    """Whether a normalised label ends with a colour."""
    return label.rsplit(" ", 1)[-1] in _COLOURS


BASIC = LabelSet(["Nothing", "Check", "Checkmate"])
FULL = LabelSet(["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"])
