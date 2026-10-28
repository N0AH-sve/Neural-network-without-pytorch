import re
from pathlib import Path

import numpy as np
import pytest

from my_torch.analyser import analyzer
from my_torch.analyser.predict import LABEL_SETS, Predict
from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import INPUT_SIZE, encode
from my_torch.encoding.labels import BASIC, FULL
from my_torch.nn_module.neural_network_module import Neural_network_module

ROOT = Path(__file__).resolve().parent.parent
CHESS_FILES = ROOT / "chess_files"
LOADING_FILES = ROOT / "loading_files"

BASIC_NET = str(LOADING_FILES / "chess_basic.nn")
COLOURED_NET = str(LOADING_FILES / "chess_coloured.nn")


def board(name: str) -> str:
    return str(CHESS_FILES / name)


def samples_of(name: str) -> list:
    with open(board(name)) as file:
        return read_samples(file)


def net_shaped(tmp_path, sizes: list[int], name: str = "net.nn") -> str:
    """Save an untrained network of the given shape and return its path."""
    path = tmp_path / name
    Neural_network_module(int_layers=sizes).save(str(path))
    return str(path)


# --- label_set_of ----------------------------------------------------------

def test_three_output_neurons_select_the_plain_states(tmp_path):
    net = Neural_network_module.load(net_shaped(tmp_path, [INPUT_SIZE, 8, 3]))
    assert Predict.label_set_of(net) is BASIC


def test_five_output_neurons_select_the_coloured_states(tmp_path):
    net = Neural_network_module.load(net_shaped(tmp_path, [INPUT_SIZE, 8, 5]))
    assert Predict.label_set_of(net) is FULL


@pytest.mark.parametrize("width", [1, 2, 4, 6, 12])
def test_any_other_width_is_rejected(tmp_path, width):
    net = Neural_network_module.load(net_shaped(tmp_path, [INPUT_SIZE, 8, width]))
    with pytest.raises(ValueError, match=f"{width} output neurons"):
        Predict.label_set_of(net)


def test_the_rejection_lists_the_accepted_widths(tmp_path):
    net = Neural_network_module.load(net_shaped(tmp_path, [INPUT_SIZE, 8, 4]))
    with pytest.raises(ValueError, match=re.escape("expected one of: 3, 5")):
        Predict.label_set_of(net)


def test_the_label_sets_are_keyed_by_their_own_length():
    """The keys must not drift from the sets they point at."""
    assert all(len(labels) == width for width, labels in LABEL_SETS.items())


# --- classify --------------------------------------------------------------

def test_classify_returns_one_index_per_board():
    net = Neural_network_module.load(BASIC_NET)
    samples = samples_of("subject.txt")
    assert Predict.classify(net, samples).shape == (len(samples),)


def test_classify_returns_integer_indices_inside_the_label_range():
    net = Neural_network_module.load(BASIC_NET)
    indices = Predict.classify(net, samples_of("unlabelled.txt"))
    assert np.issubdtype(indices.dtype, np.integer)
    assert indices.min() >= 0
    assert indices.max() < len(BASIC)


def test_classify_on_the_coloured_network_can_reach_the_last_class():
    net = Neural_network_module.load(COLOURED_NET)
    indices = Predict.classify(net, samples_of("unlabelled.txt"))
    assert indices.max() < len(FULL)


def test_classify_handles_an_empty_board_list():
    net = Neural_network_module.load(BASIC_NET)
    assert Predict.classify(net, []).shape == (0,)


def test_classify_agrees_with_one_forward_pass_per_board():
    """The batched pass is only a faster spelling of the obvious loop."""
    net = Neural_network_module.load(BASIC_NET)
    samples = samples_of("subject.txt")

    one_by_one = [net.forward(encode(s.position)).argmax() for s in samples]

    assert list(Predict.classify(net, samples)) == one_by_one


def test_classify_keeps_the_order_of_the_boards():
    net = Neural_network_module.load(BASIC_NET)
    samples = samples_of("unlabelled.txt")

    whole = list(Predict.classify(net, samples))
    reversed_back = list(Predict.classify(net, samples[::-1]))[::-1]

    assert whole == reversed_back


def test_classify_ignores_the_expected_label():
    """labelled.txt and unlabelled.txt hold the same boards."""
    net = Neural_network_module.load(BASIC_NET)
    assert list(Predict.classify(net, samples_of("labelled.txt"))) == list(
        Predict.classify(net, samples_of("unlabelled.txt"))
    )


# --- run -------------------------------------------------------------------

@pytest.mark.parametrize("name", ["subject.txt", "unlabelled.txt", "labelled.txt"])
def test_run_prints_one_line_per_board(name, capsys):
    assert Predict(BASIC_NET, board(name)).run() == 0

    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == len(samples_of(name))


def test_run_prints_the_labels_of_the_indices_it_computed(capsys):
    net = Neural_network_module.load(BASIC_NET)
    expected = [BASIC.label(i) for i in Predict.classify(net, samples_of("subject.txt"))]

    Predict(BASIC_NET, board("subject.txt")).run()

    assert capsys.readouterr().out.splitlines() == expected


def test_run_only_prints_names_of_the_networks_label_set(capsys):
    Predict(COLOURED_NET, board("unlabelled.txt")).run()
    printed = set(capsys.readouterr().out.splitlines())
    assert printed <= set(FULL.classes)


def test_run_on_an_empty_file_prints_nothing(capsys):
    assert Predict(BASIC_NET, board("empty.txt")).run() == 0
    assert capsys.readouterr().out == ""


def test_run_skips_blank_lines(capsys):
    """blank_lines.txt holds 6 lines, only 2 of which are boards."""
    assert Predict(BASIC_NET, board("blank_lines.txt")).run() == 0
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_run_accepts_boards_without_the_move_counters(capsys):
    assert Predict(BASIC_NET, board("four_fields.txt")).run() == 0
    assert len(capsys.readouterr().out.splitlines()) == 5


def test_run_accepts_boards_padded_with_spaces(capsys):
    assert Predict(BASIC_NET, board("trailing_spaces.txt")).run() == 0
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_run_ignores_an_unknown_label(capsys):
    """A label is never read in prediction mode, valid or not."""
    assert Predict(BASIC_NET, board("unknown_label.txt")).run() == 0
    assert len(capsys.readouterr().out.splitlines()) == 3


# --- run: the network is checked before the file is read -------------------

def test_run_rejects_a_network_whose_output_is_not_a_known_label_set(tmp_path):
    path = net_shaped(tmp_path, [INPUT_SIZE, 8, 4])
    with pytest.raises(ValueError, match="4 output neurons"):
        Predict(path, board("subject.txt")).run()


@pytest.mark.parametrize("size", [64, 768, 770])
def test_run_rejects_a_network_that_does_not_take_the_encoding(tmp_path, size):
    path = net_shaped(tmp_path, [size, 8, 3])
    with pytest.raises(ValueError, match=f"takes {size} inputs, expected {INPUT_SIZE}"):
        Predict(path, board("subject.txt")).run()


def test_the_network_is_checked_before_the_chess_file_is_opened(tmp_path):
    """A bad network must not be reported as a missing file."""
    path = net_shaped(tmp_path, [INPUT_SIZE, 8, 4])
    with pytest.raises(ValueError, match="4 output neurons"):
        Predict(path, str(tmp_path / "absent.txt")).run()


def test_run_rejects_a_missing_network(tmp_path):
    with pytest.raises(OSError):
        Predict(str(tmp_path / "absent.nn"), board("subject.txt")).run()


def test_run_rejects_a_missing_chess_file(tmp_path):
    with pytest.raises(OSError):
        Predict(BASIC_NET, str(tmp_path / "absent.txt")).run()


# --- run: broken boards, and the no-partial-output guarantee ---------------

@pytest.mark.parametrize("name, message", [
    ("invalid_one_field.txt", "line 2: expected 4 or 6 fields, got 1"),
    ("invalid_five_fields.txt", "expected 4 or 6 fields, got 5"),
    ("invalid_rank_too_long.txt", "rank 7 has 9 squares, expected 8"),
    ("invalid_rank_too_short.txt", "rank 7 has 7 squares, expected 8"),
    ("invalid_rank_count.txt", "expected 8 ranks, got 7"),
    ("invalid_char.txt", "invalid character 'x' in rank 8"),
    ("invalid_side.txt", "side to move: expected 'w' or 'b', got 'x'"),
    ("invalid_castling.txt", "castling: expected"),
    ("invalid_en_passant.txt", "en passant: expected"),
    ("invalid_halfmove.txt", "halfmove: expected a non-negative integer"),
    ("missing_black_king.txt", "expected exactly one 'k', got 0"),
    ("two_white_kings.txt", "expected exactly one 'K', got 2"),
])
def test_run_rejects_a_broken_board(name, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        Predict(BASIC_NET, board(name)).run()


def test_a_broken_board_leaves_nothing_on_stdout(capsys):
    """error_on_last_line.txt holds 10 valid boards then a broken one."""
    with pytest.raises(ValueError):
        Predict(BASIC_NET, board("error_on_last_line.txt")).run()

    assert capsys.readouterr().out == ""


# --- end to end through main ----------------------------------------------

@pytest.mark.parametrize("net", [BASIC_NET, COLOURED_NET])
def test_main_predicts_every_board_and_exits_zero(net, capsys):
    assert analyzer.main(["--predict", net, board("subject.txt")]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5


def test_main_turns_a_broken_board_into_84(capsys):
    argv = ["--predict", BASIC_NET, board("invalid_char.txt")]
    assert analyzer.main(argv) == 84

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("my_torch_analyzer: ")


def test_main_turns_a_mismatched_network_into_84(tmp_path, capsys):
    path = net_shaped(tmp_path, [INPUT_SIZE, 8, 4])
    assert analyzer.main(["--predict", path, board("subject.txt")]) == 84
    assert "4 output neurons" in capsys.readouterr().err


def test_main_reports_the_swapped_arguments_as_an_error(capsys):
    """A chess file where the network belongs cannot be loaded."""
    assert analyzer.main(["--predict", board("subject.txt"), BASIC_NET]) == 84
    assert capsys.readouterr().err.startswith("my_torch_analyzer: ")
