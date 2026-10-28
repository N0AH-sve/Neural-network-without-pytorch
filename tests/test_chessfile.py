import pytest

from my_torch.encoding.chessfile import parse_line
from my_torch.encoding.position import FenError, Position

KINGS = "4k3/8/8/8/8/8/8/4K3 w - -"

# The subject's boards with the answers of its "full" network.
SUBJECT_LINES = [
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "Checkmate Black"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Nothing"),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "Nothing"),
    ("rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3", "Check White"),
    ("8/8/8/8/8/8/8/k1K5 w - - 0 1", "Nothing"),
]


@pytest.mark.parametrize("fen, label", SUBJECT_LINES)
def test_subject_lines(fen, label):
    sample = parse_line(f"{fen} {label}")
    assert sample.position == Position.from_fen(fen)
    assert sample.label == label


@pytest.mark.parametrize("fen, _", SUBJECT_LINES)
def test_label_is_optional(fen, _):
    assert parse_line(fen).label is None


@pytest.mark.parametrize("label", ["Nothing", "Check", "Checkmate", "Check White", "Checkmate Black"])
def test_one_or_two_word_labels(label):
    assert parse_line(f"{KINGS} 0 1 {label}").label == label


def test_label_after_a_4_field_fen():
    sample = parse_line(f"{KINGS} Checkmate White")
    assert (sample.position.halfmove, sample.position.fullmove) == (0, 1)
    assert sample.label == "Checkmate White"


def test_counters_are_not_mistaken_for_a_label():
    sample = parse_line(f"{KINGS} 3 42")
    assert (sample.position.halfmove, sample.position.fullmove) == (3, 42)
    assert sample.label is None


@pytest.mark.parametrize("line", [
    f"{KINGS} 0 1 Check White\n",
    f"{KINGS} 0 1 Check White\r\n",
    f"  {KINGS}   0  1   Check \t White  ",
])
def test_whitespace_is_normalised(line):
    assert parse_line(line).label == "Check White"


@pytest.mark.parametrize("line, message", [
    (f"{KINGS} 0 Check", "expected 4 or 6 fields"),
    (f"{KINGS} -1 1 Check", "halfmove"),
    ("", "expected 4 or 6 fields"),
    ("Check", "expected 4 or 6 fields"),
])
def test_invalid_lines(line, message):
    with pytest.raises(FenError, match=message):
        parse_line(line)
