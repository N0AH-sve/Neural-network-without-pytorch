"""Chessboards read from Forsyth-Edwards Notation (FEN)."""

from dataclasses import dataclass
from typing import Literal, Self

PIECES = "PNBRQKpnbrqk"


class FenError(ValueError):
    """Raised when a chessboard is not valid FEN."""


@dataclass(frozen=True)
class Position:
    board: tuple[str, ...]
    side: Literal["w", "b"]
    castling: str
    en_passant: str
    halfmove: int = 0
    fullmove: int = 1

    @classmethod
    def from_fen(cls, fen: str) -> Self:
        """Build a position from 4 FEN fields, or 6 with the move counters."""
        fields = fen.split()
        if len(fields) not in (4, 6):
            raise FenError(f"expected 4 or 6 fields, got {len(fields)}")
        placement, side, castling, en_passant = fields[:4]
        counters = [int(counter) for counter in fields[4:]]
        return cls(_parse_board(placement), side, castling, en_passant, *counters)


def _parse_board(placement: str) -> tuple[str, ...]:
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise FenError(f"board: expected 8 ranks, got {len(ranks)}")
    board = tuple(_parse_rank(rank, 8 - row) for row, rank in enumerate(ranks))
    _check_kings(board)
    return board


def _parse_rank(rank: str, number: int) -> str:
    squares = ""
    for char in rank:
        if char in "12345678":
            squares += "." * int(char)
        elif char in PIECES:
            squares += char
        else:
            raise FenError(f"board: invalid character {char!r} in rank {number}")
    if len(squares) != 8:
        raise FenError(f"board: rank {number} has {len(squares)} squares, expected 8")
    return squares


def _check_kings(board: tuple[str, ...]) -> None:
    for king in "Kk":
        count = sum(rank.count(king) for rank in board)
        if count != 1:
            raise FenError(f"board: expected exactly one {king!r}, got {count}")
