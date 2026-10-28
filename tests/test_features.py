import numpy as np
import pytest

from my_torch.encoding.features import INPUT_SIZE, encode
from my_torch.encoding.position import Position

START = Position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


def index(plane, square):
    """Index of a piece plane (0 = P ... 11 = k) on a square like 'e1'."""
    col = ord(square[0]) - ord("a")
    row = 8 - int(square[1])
    return plane * 64 + row * 8 + col


def test_input_size_is_769():
    assert INPUT_SIZE == 12 * 64 + 1 == 769


def test_shape_and_values():
    x = encode(START)
    assert x.shape == (INPUT_SIZE,)
    assert x.dtype == np.float64
    assert set(np.unique(x)) <= {0.0, 1.0}


def test_one_value_per_piece():
    assert encode(START)[:-1].sum() == 32


@pytest.mark.parametrize("plane, square", [
    (5, "e1"),    # white king
    (11, "e8"),   # black king
    (4, "d1"),    # white queen
    (3, "a1"),    # white rook, first square of rank 1
    (9, "h8"),    # black rook, last square of rank 8
    (0, "e2"),    # white pawn
    (6, "e7"),    # black pawn
])
def test_pieces_land_on_their_plane_and_square(plane, square):
    assert encode(START)[index(plane, square)] == 1.0


def test_corners_of_the_board():
    """a8 is index 0 of a plane and h1 is index 63."""
    position = Position.from_fen("k7/8/8/8/8/8/8/7K w - -")
    assert np.flatnonzero(encode(position)[:-1]).tolist() == [5 * 64 + 63, 11 * 64 + 0]


def test_side_to_move_is_the_last_value():
    white = encode(Position.from_fen("4k3/8/8/8/8/8/8/4K3 w - -"))
    black = encode(Position.from_fen("4k3/8/8/8/8/8/8/4K3 b - -"))
    assert white[-1] == 1.0
    assert black[-1] == 0.0
    assert np.array_equal(white[:-1], black[:-1])


def test_other_fields_are_not_encoded():
    a = encode(Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
    b = encode(Position.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 40 90"))
    assert np.array_equal(a, b)
