"""Lines of a chess file: a FEN board, optionally followed by its label."""

from dataclasses import dataclass

from my_torch.encoding.position import Position


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
