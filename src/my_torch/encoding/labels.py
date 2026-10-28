"""The game states a network predicts, mapped to its output neurons."""

from collections.abc import Iterable


class LabelSet:
    """Class names in the order of the output neurons."""

    def __init__(self, classes: Iterable[str]):
        self.classes = tuple(classes)
        self._indices = {_normalise(name): i for i, name in enumerate(self.classes)}
        if len(self._indices) != len(self.classes):
            raise ValueError(f"duplicate class in {self.classes}")

    def __len__(self) -> int:
        return len(self.classes)

    def index(self, label: str) -> int:
        """Output neuron of a label, ignoring case and extra spaces."""
        try:
            return self._indices[_normalise(label)]
        except KeyError:
            valid = ", ".join(self.classes)
            raise ValueError(f"unknown label {label!r}, expected one of: {valid}") from None

    def label(self, index: int) -> str:
        return self.classes[index]


def _normalise(label: str) -> str:
    return " ".join(label.split()).lower()
