"""Multiply a train / validation split by the symmetries of chess.

    uv run python -m my_torch.tools.augment_dataset [-o OUTDIR] [--seed N]
        [TRAIN [VALIDATION]]

Every board gives up to four boards with a label known for sure:

- itself;
- its mirror, files a <-> h: check and mate do not care about left and right
  (castling rights are dropped, they would point at the wrong rooks);
- its colour swap, ranks 1 <-> 8 with White and Black exchanged and the other
  side to move: pawns still move forward, and the colour of the label swaps;
- both, which is the board turned by 180 degrees with the colours swapped.

A plain rotation is not one of them: pawns would move backwards.

Both files are augmented apart, so the copies of a validation board never
land in the training file; a training board that turns into a validation
board is dropped too. A board (placement + side to move) is kept once.
"""

import argparse
import random
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.labels import FULL

DEFAULT_INPUT = Path("dataset/mixed")
DEFAULT_OUTPUT = Path("dataset/augmented")
_SWAP_COLOUR = {"white": "Black", "black": "White"}


def _expand(placement: str) -> list[str]:
    """Ranks from 8 to 1, one character per square, '.' when empty."""
    return ["".join("." * int(c) if c.isdigit() else c for c in rank)
            for rank in placement.split("/")]


def _compress(ranks: Iterable[str]) -> str:
    out = []
    for rank in ranks:
        text, empty = "", 0
        for square in rank:
            if square == ".":
                empty += 1
                continue
            text += (str(empty) if empty else "") + square
            empty = 0
        out.append(text + (str(empty) if empty else ""))
    return "/".join(out)


def mirror(fields: list[str], label: str) -> tuple[list[str], str]:
    """Files a <-> h; the label does not change."""
    placement, side, _, en_passant, *counters = fields
    if en_passant != "-":
        en_passant = "abcdefgh"[7 - "abcdefgh".index(en_passant[0])] + en_passant[1]
    ranks = [rank[::-1] for rank in _expand(placement)]
    return [_compress(ranks), side, "-", en_passant, *counters], label


def swap_colours(fields: list[str], label: str) -> tuple[list[str], str]:
    """Ranks 1 <-> 8, White <-> Black, other side to move, other colour."""
    placement, side, castling, en_passant, *counters = fields
    ranks = [rank.swapcase() for rank in reversed(_expand(placement))]
    if castling != "-":
        castling = "".join(right for right in "KQkq" if right.swapcase() in castling)
    if en_passant != "-":
        en_passant = en_passant[0] + {"3": "6", "6": "3"}[en_passant[1]]
    words = label.split()
    if words[-1].lower() in _SWAP_COLOUR:
        words[-1] = _SWAP_COLOUR[words[-1].lower()]
    return ([_compress(ranks), "b" if side == "w" else "w", castling, en_passant, *counters],
            " ".join(words))


def variants(line: str) -> Iterator[str]:
    """The line and its three symmetric copies; a FEN without counters keeps
    none."""
    words = line.split()
    count = 6 if len(words) > 6 and words[4].isdecimal() else 4
    fields, label = words[:count], " ".join(words[count:])
    mirrored = mirror(fields, label)
    for board, text in ((fields, label), mirrored, swap_colours(fields, label),
                        swap_colours(*mirrored)):
        yield f"{' '.join(board)} {text}"


def _key(line: str) -> tuple[str, str]:
    words = line.split()
    return words[0], words[1]


def augment(lines: Iterable[str], exclude: set | None = None) -> tuple[list[str], set]:
    """Every variant of every line, once, leaving out the boards of exclude;
    also returns the boards kept."""
    seen = set(exclude or ())
    kept_keys = set()
    out = []
    for line in lines:
        if not line.strip():
            continue
        for variant in variants(line):
            key = _key(variant)
            if key in seen:
                continue
            seen.add(key)
            kept_keys.add(key)
            out.append(variant)
    return out, kept_keys


def _report(path: Path, lines: list[str]) -> str:
    samples = read_samples(lines, require_labels=True)
    counts = Counter(FULL.label(FULL.index(s.label, s.position.side)) for s in samples)
    classes = ", ".join(f"{counts[name]} {name}" for name in FULL.classes)
    return f"{path}: {len(lines)} boards ({classes})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m my_torch.tools.augment_dataset",
        description="Multiply a train / validation split by the symmetries of chess.")
    parser.add_argument("train", nargs="?", type=Path, default=DEFAULT_INPUT / "train.txt",
                        help="training file (default: dataset/mixed/train.txt)")
    parser.add_argument("validation", nargs="?", type=Path,
                        default=DEFAULT_INPUT / "validation.txt",
                        help="validation file (default: dataset/mixed/validation.txt)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, type=Path,
                        help="folder for train.txt and validation.txt (default: dataset/augmented)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    args = parser.parse_args(argv)

    try:
        with open(args.validation) as file:
            validation, validation_keys = augment(file)
        with open(args.train) as file:
            train, _ = augment(file, exclude=validation_keys)
        rng = random.Random(args.seed)
        rng.shuffle(train)
        rng.shuffle(validation)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, lines in (("train.txt", train), ("validation.txt", validation)):
            (args.output / name).write_text("".join(f"{line}\n" for line in lines))
        reports = [_report(args.output / name, lines)
                   for name, lines in (("train.txt", train), ("validation.txt", validation))]
    except (OSError, ValueError) as error:
        print(f"augment_dataset: {error}", file=sys.stderr)
        return 84

    print("\n".join(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
