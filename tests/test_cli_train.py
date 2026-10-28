"""my_torch_analyzer --train, run end to end through main()."""

import pytest

from my_torch.analyser import analyzer
from my_torch.encoding.features import INPUT_SIZE
from my_torch.encoding.labels import BASIC
from my_torch.nn_module.neural_network_module import Neural_network_module

TRAINING_LINES = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White",
]


@pytest.fixture
def files(tmp_path):
    network = tmp_path / "net.nn"
    Neural_network_module([INPUT_SIZE, 8, len(BASIC)], epochs=5).save(str(network))
    boards = tmp_path / "boards.txt"
    boards.write_text("".join(f"{line}\n" for line in TRAINING_LINES))
    return network, boards


def test_train_exits_0_and_reports(files, capsys):
    network, boards = files
    assert analyzer.main(["--train", str(network), str(boards)]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("Number of boards : 3, ")
    assert err == ""


def test_train_with_save(tmp_path, files):
    network, boards = files
    savefile = tmp_path / "out.nn"
    original = network.read_bytes()
    assert analyzer.main(["--train", "--save", str(savefile), str(network), str(boards)]) == 0
    assert network.read_bytes() == original
    assert Neural_network_module.load(str(savefile)).layers[0].input == INPUT_SIZE


@pytest.mark.parametrize("make_argv, message", [
    (lambda net, boards: ["--train", net], "got 1 file"),
    (lambda net, boards: ["--train", "--predict", net, boards], "exactly one of"),
    (lambda net, boards: ["--predict", "--save", "out.nn", net, boards], "only works with --train"),
    (lambda net, boards: ["--train", "missing.nn", boards], "missing.nn"),
    (lambda net, boards: ["--train", net, "missing.txt"], "missing.txt"),
])
def test_errors_exit_84_on_stderr(files, capsys, make_argv, message):
    network, boards = files
    assert analyzer.main(make_argv(str(network), str(boards))) == 84
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("my_torch_analyzer: ")
    assert message in err


def test_unlabelled_board_exits_84(tmp_path, files, capsys):
    network, _ = files
    boards = tmp_path / "unlabelled.txt"
    boards.write_text(f"{TRAINING_LINES[0]}\n4k3/8/8/8/8/8/8/4K3 w - -\n")
    assert analyzer.main(["--train", str(network), str(boards)]) == 84
    assert "line 2: missing the expected output" in capsys.readouterr().err


def test_network_that_does_not_fit_exits_84(tmp_path, files, capsys):
    _, boards = files
    network = tmp_path / "small.nn"
    Neural_network_module([4, 3, 2]).save(str(network))
    assert analyzer.main(["--train", str(network), str(boards)]) == 84
    assert "takes 4 inputs, expected 769" in capsys.readouterr().err
