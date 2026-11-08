"""Mix chess files into shuffled train and validation files.

    uv run python -m my_torch.tools.shuffle_dataset [-o OUTDIR] [--seed N]
        [--validation RATIO] [--limit N] [INPUT ...]

Every line is parsed, so a broken file fails here rather than mid-training.
A board (placement + side to move) is kept once, so the same position cannot
end up in both files. The shuffle is seeded: the same command gives the same
split. Boards are counted per coloured class, the colour of a label that does
not carry one being read from the side to move.
"""

import argparse
import random
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.labels import FULL

DEFAULT_OUTPUT = Path("dataset/mixed")

# (line, coloured class) for each board: a label given without a colour is
# completed from the side to move, so the report shows the real balance
# whether the files name the colours or not.
Entry = tuple[str, str]


def collect_files(inputs: Iterable[str], output: Path) -> list[Path]:
    """Files as given; folders give their .txt files, except earlier outputs.

    Both output and DEFAULT_OUTPUT are skipped: reading an earlier train.txt
    would change the order before the shuffle, and so the split.
    """
    skipped = {output.resolve(), DEFAULT_OUTPUT.resolve()}
    files = []
    for name in inputs:
        path = Path(name)
        if path.is_dir():
            files.extend(file for file in sorted(path.rglob("*.txt"))
                         if skipped.isdisjoint(file.resolve().parents))
        else:
            files.append(path)
    return files


def read_entries(files: Iterable[Path]) -> tuple[list[Entry], int]:
    """Labelled lines of every file, without duplicates, and how many were dropped."""
    seen = set()
    entries = []
    duplicates = 0
    for path in files:
        with open(path) as file:
            lines = [" ".join(line.split()) for line in file]
        try:
            samples = read_samples(lines, require_labels=True)
            classes = [FULL.label(FULL.index(sample.label, sample.position.side))
                       for sample in samples]
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        for line, sample, label in zip(filter(None, lines), samples, classes, strict=True):
            key = (sample.position.board, sample.position.side)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            entries.append((line, label))
    return entries, duplicates


def split(entries: list[Entry], validation: float, seed: int,
          limit: int | None = None) -> tuple[list[Entry], list[Entry]]:
    """Shuffle, keep the first limit entries, then cut off the validation part."""
    entries = list(entries)
    random.Random(seed).shuffle(entries)
    if limit is not None:
        entries = entries[:limit]
    size = round(len(entries) * validation)
    return entries[size:], entries[:size]


def _write(path: Path, entries: list[Entry]) -> None:
    path.write_text("".join(f"{line}\n" for line, _ in entries))


def _report(path: Path, entries: list[Entry]) -> str:
    counts = Counter(label for _, label in entries)
    classes = ", ".join(f"{counts[name]} {name}" for name in FULL.classes)
    return f"{path}: {len(entries)} boards ({classes})"


def _ratio(value: str) -> float:
    ratio = float(value)
    if not 0 <= ratio < 1:
        raise argparse.ArgumentTypeError(f"expected a ratio in [0, 1), got {value}")
    return ratio


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m my_torch.tools.shuffle_dataset",
        description="Mix chess files into shuffled train and validation files.")
    parser.add_argument("inputs", nargs="*", default=["dataset"],
                        help="chess files or folders of .txt files (default: dataset)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, type=Path,
                        help="folder for train.txt and validation.txt (default: dataset/mixed)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    parser.add_argument("--validation", type=_ratio, default=0.2,
                        help="share of the boards kept for validation (default: 0.2)")
    parser.add_argument("--limit", type=_positive,
                        help="keep only this many boards, picked at random")
    args = parser.parse_args(argv)

    try:
        entries, duplicates = read_entries(collect_files(args.inputs, args.output))
        if not entries:
            raise ValueError("no board found")
        train, validation = split(entries, args.validation, args.seed, args.limit)
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output / "train.txt", train)
        _write(args.output / "validation.txt", validation)
    except (OSError, ValueError) as error:
        print(f"shuffle_dataset: {error}", file=sys.stderr)
        return 84

    print(f"{len(entries)} boards read, {duplicates} duplicates removed")
    print(_report(args.output / "train.txt", train))
    print(_report(args.output / "validation.txt", validation))
    return 0


if __name__ == "__main__":
    sys.exit(main())
