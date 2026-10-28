"""Lines of a chess file: a FEN board, optionally followed by its label."""

from collections.abc import Iterable
from dataclasses import dataclass

from my_torch.encoding.position import FenError, Position


@dataclass(frozen=True)
class Sample:
    position: Position
    label: str | None


def parse_line(line: str) -> Sample:
    """Split '<FEN> [label]', the label being one or more words.

    The label starts at the first token after the 4 mandatory FEN fields that
    begins with a letter: labels are words, move counters are numbers.
    """
    tokens = line.split()
    end = next(
        (i for i in range(4, len(tokens)) if tokens[i][0].isalpha()),
        len(tokens),
    )
    position = Position.from_fen(" ".join(tokens[:end]))
    return Sample(position, " ".join(tokens[end:]) or None)


def read_samples(lines: Iterable[str], *, require_labels: bool = False) -> list[Sample]:
    """Parse every non-blank line, e.g. of an open chess file.

    With require_labels, as in train mode, a line without a label is an error.
    """
    samples = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            sample = parse_line(line)
        except FenError as error:
            raise FenError(f"line {number}: {error}") from error
        if require_labels and sample.label is None:
            raise FenError(f"line {number}: missing the expected output")
        samples.append(sample)
    return samples
