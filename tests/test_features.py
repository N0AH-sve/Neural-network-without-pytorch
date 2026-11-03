import copy

import numpy as np
import pytest

from my_torch.encoding.features import INPUT_SIZE, encode, encode_all
from my_torch.encoding.position import Position
from my_torch.nn_module.neural_network_module import Neural_network_module

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
    assert x.dtype == np.bool_


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


def test_encode_all_stacks_one_row_per_position():
    positions = [
        START,
        Position.from_fen("4k3/8/8/8/8/8/8/4K3 b - -"),
        Position.from_fen("k7/8/8/8/8/8/8/7K w - -"),
    ]
    x = encode_all(positions)
    assert x.shape == (3, INPUT_SIZE)
    for row, position in zip(x, positions, strict=True):
        assert np.array_equal(row, encode(position))


def test_encode_all_accepts_a_generator():
    assert encode_all(START for _ in range(4)).shape == (4, INPUT_SIZE)


def test_encode_all_of_nothing_is_an_empty_matrix():
    assert encode_all([]).shape == (0, INPUT_SIZE)


def test_encode_all_is_boolean():
    assert encode_all([START, START]).dtype == np.bool_
    assert encode_all([]).dtype == np.bool_


def test_a_board_takes_one_byte_per_value():
    """8 times less than float64: 769 bytes per board instead of 6,152."""
    assert encode_all([START, START]).nbytes == 2 * INPUT_SIZE


def test_the_network_reads_booleans_like_floats():
    x = encode_all([START, Position.from_fen("k7/8/8/8/8/8/8/7K b - -")])
    network = Neural_network_module([INPUT_SIZE, 8, 3])
    assert np.array_equal(network.forward(x), network.forward(x.astype(float)))


def test_training_on_booleans_matches_training_on_floats():
    x = encode_all([START, Position.from_fen("k7/8/8/8/8/8/8/7K b - -")] * 5)
    y = np.eye(3)[[0, 1] * 5]
    network = Neural_network_module([INPUT_SIZE, 8, 3], epochs=3, batch_size=4)
    twin = copy.deepcopy(network)
    network.train(x, y)
    twin.train(x.astype(float), y)
    for trained, expected in zip(network.layers, twin.layers, strict=True):
        assert np.array_equal(trained.w, expected.w)
        assert np.array_equal(trained.b, expected.b)
