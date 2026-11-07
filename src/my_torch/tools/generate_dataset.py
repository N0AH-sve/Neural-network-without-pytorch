"""Generate labelled chess boards, the raw material shuffle_dataset mixes.

    uv run --with chess python -m my_torch.tools.generate_dataset \
        [-o OUTDIR] [--boards N] [--seed N] [--bias RATIO]
    uv run python -m my_torch.tools.shuffle_dataset   # then: dataset/mixed/

python-chess is NOT a dependency of this project: nothing in src/ imports it
but this file, which no module imports either. `uv run --with chess` installs
it for the length of the command and leaves pyproject.toml and uv.lock alone.

Boards come from random self-play. Pure random play almost never reaches
check, let alone mate, so when a quota still needs them the generator plays a
mating move if one exists, then a checking one, with probability --bias. Each
of the five coloured states gets the same quota, so the file does not come out
as a heap of "Nothing": a 3-output network reads the same boards with the
colour dropped, so one file serves both widths.

The seed makes a run reproducible. Duplicate boards are skipped here, and
shuffle_dataset drops them again across files.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT = Path("dataset/generated")
STATES = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
MAX_PLIES = 200


def label_of(chess, board) -> str:
    """The state of a board, in the project's convention: the colour names the
    side giving check or mate, which is the side that has just played."""
    if not board.is_check():
        return "Nothing"
    colour = chess.COLOR_NAMES[not board.turn].capitalize()
    return f"{'Checkmate' if board.is_checkmate() else 'Check'} {colour}"


def _forcing_move(board, legal: list, rng: random.Random, want_mate: bool):
    """A move giving mate, or just check, or None. Mate implies check, so only
    the checking moves are ever pushed, which keeps this affordable."""
    checking = [move for move in legal if board.gives_check(move)]
    if not checking:
        return None
    if want_mate:
        for move in checking:
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return move
    return rng.choice(checking)


def _next_move(board, legal: list, rng: random.Random, needs: Counter, bias: float):
    """The move to play: a forcing one when a quota still wants check or mate
    and the dice say so, a random legal one otherwise."""
    wants_mate = needs["Checkmate White"] > 0 or needs["Checkmate Black"] > 0
    wants_check = wants_mate or needs["Check White"] > 0 or needs["Check Black"] > 0
    if wants_check and rng.random() < bias:
        forcing = _forcing_move(board, legal, rng, wants_mate)
        if forcing is not None:
            return forcing
    return rng.choice(legal)


def generate(chess, quota: int, seed: int, bias: float) -> list[str]:
    """quota boards of each of the five states, as '<FEN> <label>' lines."""
    rng = random.Random(seed)
    needs = Counter({state: quota for state in STATES})
    seen = set()
    lines = []
    while any(count > 0 for count in needs.values()):
        board = chess.Board()
        for _ in range(MAX_PLIES):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(_next_move(board, legal, rng, needs, bias))
            label = label_of(chess, board)
            key = board.board_fen(), board.turn
            if needs[label] <= 0 or key in seen:
                continue
            seen.add(key)
            needs[label] -= 1
            lines.append(f"{board.fen()} {label}")
            if not any(count > 0 for count in needs.values()):
                break
    rng.shuffle(lines)
    return lines


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def _ratio(value: str) -> float:
    ratio = float(value)
    if not 0 <= ratio <= 1:
        raise argparse.ArgumentTypeError(f"expected a ratio in [0, 1], got {value}")
    return ratio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uv run --with chess python -m my_torch.tools.generate_dataset",
        description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, type=Path,
                        help=f"folder for boards.txt (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--boards", type=_positive, default=200_000,
                        help="boards to generate, split evenly between the five "
                             "states (default: 200000)")
    parser.add_argument("--seed", type=int, default=42, help="play seed (default: 42)")
    parser.add_argument("--bias", type=_ratio, default=0.5,
                        help="chance of forcing check when a quota wants it "
                             "(default: 0.5)")
    args = parser.parse_args(argv)

    try:
        import chess
    except ImportError:
        print("generate_dataset: python-chess is missing. Run this through\n"
              "    uv run --with chess python -m my_torch.tools.generate_dataset",
              file=sys.stderr)
        return 84

    quota = max(1, args.boards // len(STATES))
    try:
        lines = generate(chess, quota, args.seed, args.bias)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "boards.txt"
        path.write_text("".join(f"{line}\n" for line in lines))
    except OSError as error:
        print(f"generate_dataset: {error}", file=sys.stderr)
        return 84

    counts = Counter(line.split(" ", 6)[6] for line in lines)
    print(f"{path}: {len(lines)} boards")
    for state in STATES:
        print(f"  {counts[state]:>7} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
