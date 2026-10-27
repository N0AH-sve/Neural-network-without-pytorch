import pytest

from my_torch.encoding.position import FenError, Position

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# The five boards of the subject.
SUBJECT_BOARDS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]


def test_start_position_board():
    assert Position.from_fen(START).board == (
        "rnbqkbnr",
        "pppppppp",
        "........",
        "........",
        "........",
        "........",
        "PPPPPPPP",
        "RNBQKBNR",
    )


def test_fields_after_the_board_are_kept():
    position = Position.from_fen(SUBJECT_BOARDS[2])
    assert position.side == "b"
    assert position.castling == "KQkq"
    assert position.en_passant == "d3"
    assert (position.halfmove, position.fullmove) == (0, 1)


def test_first_rank_of_the_fen_is_rank_8():
    """a1 is the last row, first column."""
    assert Position.from_fen(SUBJECT_BOARDS[4]).board[7] == "k.K....."


@pytest.mark.parametrize("fen", SUBJECT_BOARDS)
def test_subject_boards_are_8_by_8(fen):
    board = Position.from_fen(fen).board
    assert len(board) == 8
    assert all(len(rank) == 8 for rank in board)


@pytest.mark.parametrize("fen", [
    "4k3/8/8/8/8/8/8/4K3 w -",
    "4k3/8/8/8/8/8/8/4K3 w - - 0",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
    "",
])
def test_wrong_number_of_fields(fen):
    with pytest.raises(FenError, match="expected 4 or 6 fields"):
        Position.from_fen(fen)


@pytest.mark.parametrize("placement", [
    "4k3/8/8/8/8/8/4K3",
    "4k3/8/8/8/8/8/8/8/4K3",
])
def test_wrong_number_of_ranks(placement):
    with pytest.raises(FenError, match="expected 8 ranks"):
        Position.from_fen(f"{placement} w - -")


@pytest.mark.parametrize("placement", [
    "4k4/8/8/8/8/8/8/4K3",
    "4k2/8/8/8/8/8/8/4K3",
    "4k3/8/8/8/8/8/8/4KR3",
    "4k3/8/8//8/8/8/4K3",
])
def test_rank_must_have_8_squares(placement):
    with pytest.raises(FenError, match="squares, expected 8"):
        Position.from_fen(f"{placement} w - -")


@pytest.mark.parametrize("placement", [
    "4k3/8/8/8/8/8/8/4K2x",
    "4k3/8/8/8/8/8/8/4K029",
    "4k3/8/8/8/8/8/8/4K9",
])
def test_invalid_board_character(placement):
    with pytest.raises(FenError, match="invalid character"):
        Position.from_fen(f"{placement} w - -")


def test_error_names_the_rank():
    with pytest.raises(FenError, match="rank 1"):
        Position.from_fen("4k3/8/8/8/8/8/8/4K2x w - -")
