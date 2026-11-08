"""Generate labelled chess boards, the raw material shuffle_dataset mixes.

    uv run --with chess python -m my_torch.tools.generate_dataset \
        [-o OUTDIR] [--boards N] [--seed N] [--jobs N]
    uv run python -m my_torch.tools.shuffle_dataset   # then: dataset/mixed/

python-chess is NOT a dependency of this project: nothing in src/ imports it
but this file, which no module imports either. `uv run --with chess` installs
it for the length of the command and leaves pyproject.toml and uv.lock alone.

Every board comes from one same source: a random sparse position, two kings
plus a handful of pieces. That matters more than it looks. Random self-play
reaches checkmate about 500 times more rarely than it reaches a quiet board,
so a dataset that played its way to mate would have to draw its mates from
somewhere else -- and the classes would then differ by how many pieces they
hold, which a network learns instead of the chess. Drawing all five states
from the same positions keeps that shortcut shut: the states differ by the
position, not by the way it was built.

From one sparse position the generator harvests the board as it stands, and
the boards after each checking move, which is where the mates come from.

The seed makes a run reproducible, jobs apart: each worker seeds itself from
it. Duplicate boards are dropped here and again by shuffle_dataset.
"""

import argparse
import os
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

DEFAULT_OUTPUT = Path("dataset/generated")
STATES = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
PIECE_NAMES = ("q", "r", "r", "b", "n", "p")
MIN_PIECES, MAX_PIECES = 2, 6
STALL_LIMIT = 400_000


def label_of(chess, board) -> str:
    """The state of a board, in the project's convention: the colour names the
    side giving check or mate, which is the side that has just played."""
    if not board.is_check():
        return "Nothing"
    colour = chess.COLOR_NAMES[not board.turn].capitalize()
    return f"{'Checkmate' if board.is_checkmate() else 'Check'} {colour}"


def sparse_position(chess, rng: random.Random, attacker: bool):
    """A random legal position with `attacker` to move, or None.

    Two kings far enough apart, then a few pieces on free squares, mostly the
    attacker's. is_valid rejects what the rules cannot hold: a pawn on the
    back rank, the side not to move already in check, kings side by side.
    """
    board = chess.Board(None)
    squares = rng.sample(range(64), 2 + rng.randint(MIN_PIECES, MAX_PIECES))
    white_king, black_king = squares[0], squares[1]
    if chess.square_distance(white_king, black_king) < 2:
        return None
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    for square in squares[2:]:
        colour = attacker if rng.random() < 0.7 else not attacker
        name = rng.choice(PIECE_NAMES)
        if name == "p" and chess.square_rank(square) in (0, 7):
            name = "r"
        board.set_piece_at(square, chess.Piece.from_symbol(
            name.upper() if colour == chess.WHITE else name))
    board.turn = attacker
    return board if board.is_valid() else None


def _harvest(chess, board, needs: Counter, seen: set, lines: list) -> int:
    """Record the board and every board one checking move away, as far as the
    quotas still want them; returns how many were kept."""
    kept = 0
    for candidate in _reachable(chess, board):
        label = label_of(chess, candidate)
        if needs[label] <= 0:
            continue
        key = candidate.board_fen(), candidate.turn
        if key in seen:
            continue
        seen.add(key)
        needs[label] -= 1
        lines.append(f"{candidate.fen()} {label}")
        kept += 1
    return kept


def _reachable(chess, board):
    """The board itself, then the board after each move giving check: mate
    implies check, so the checking moves are the only ones worth pushing."""
    yield board
    for move in list(board.legal_moves):
        if not board.gives_check(move):
            continue
        board.push(move)
        yield board.copy(stack=False)
        board.pop()


def _share(index: int, jobs: int, quota: int) -> int:
    """This worker's part of a per-class quota, the remainder spread over the
    first workers so the parts add up to the whole."""
    return quota // jobs + (1 if index < quota % jobs else 0)


def _run_worker(task) -> list[str]:
    index, seed, quota, jobs = task
    import chess

    rng = random.Random(seed + index)
    needs = Counter({state: _share(index, jobs, quota) for state in STATES})
    seen: set = set()
    lines: list[str] = []
    stalled = 0
    while any(count > 0 for count in needs.values()) and stalled < STALL_LIMIT:
        board = sparse_position(chess, rng, rng.choice([chess.WHITE, chess.BLACK]))
        if board is None:
            stalled += 1
            continue
        stalled = 0 if _harvest(chess, board, needs, seen, lines) else stalled + 1
    return lines


def generate(quota: int, seed: int, jobs: int) -> list[str]:
    """quota boards of each of the five states, shuffled, without duplicates."""
    tasks = [(i, seed, quota, jobs) for i in range(jobs)]
    if jobs == 1:
        parts = [_run_worker(tasks[0])]
    else:
        with Pool(jobs) as pool:
            parts = pool.map(_run_worker, tasks)
    seen: set = set()
    lines = []
    for part in parts:
        for line in part:
            board = line.split(" ", 1)[0]
            if board in seen:
                continue
            seen.add(board)
            lines.append(line)
    random.Random(seed).shuffle(lines)
    return lines


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uv run --with chess python -m my_torch.tools.generate_dataset",
        description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, type=Path,
                        help=f"folder for boards.txt (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--boards", type=_positive, default=200_000,
                        help="boards to generate, split evenly between the five "
                             "states (default: 200000)")
    parser.add_argument("--seed", type=int, default=42, help="seed (default: 42)")
    parser.add_argument("--jobs", type=_positive, default=os.cpu_count() or 1,
                        help="worker processes (default: every core)")
    args = parser.parse_args(argv)

    try:
        import chess  # noqa: F401
    except ImportError:
        print("generate_dataset: python-chess is missing. Run this through\n"
              "    uv run --with chess python -m my_torch.tools.generate_dataset",
              file=sys.stderr)
        return 84

    quota = max(1, args.boards // len(STATES))
    try:
        lines = generate(quota, args.seed, args.jobs)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "boards.txt"
        path.write_text("".join(f"{line}\n" for line in lines))
    except OSError as error:
        print(f"generate_dataset: {error}", file=sys.stderr)
        return 84

    counts = Counter(line.split(" ", 6)[6] for line in lines)
    print(f"{path}: {len(lines)} boards ({args.jobs} workers)")
    for state in STATES:
        print(f"  {counts[state]:>8} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
