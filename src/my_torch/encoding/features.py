"""Positions turned into the input vector of the network."""

from collections.abc import Iterable

import numpy as np

from my_torch.encoding.position import PIECES, Position

_PLANE = {piece: plane for plane, piece in enumerate(PIECES)}

INPUT_SIZE = len(PIECES) * 64 + 1


def encode(position: Position) -> np.ndarray:
    """Value at plane * 64 + row * 8 + col is 1 when that piece is on that square.

    Planes follow PIECES (P N B R Q K p n b r q k); row 0 is rank 8, col 0 is
    file a. The last value is 1 when White is to move. Castling, en passant and
    the counters are left out: they do not change whether a king is in check.
    """
    x = np.zeros(INPUT_SIZE)
    for row, rank in enumerate(position.board):
        for col, square in enumerate(rank):
            if square != ".":
                x[_PLANE[square] * 64 + row * 8 + col] = 1.0
    x[-1] = float(position.side == "w")
    return x


def encode_all(positions: Iterable[Position]) -> np.ndarray:
    """Encode positions as the rows of an (N, INPUT_SIZE) matrix."""
    rows = [encode(position) for position in positions]
    return np.stack(rows) if rows else np.empty((0, INPUT_SIZE))
