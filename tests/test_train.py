import re

import numpy as np
import pytest

from my_torch.analyser.arguments import Arguments
from my_torch.analyser.train import train
from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import INPUT_SIZE, encode_all
from my_torch.encoding.labels import BASIC, FULL, for_outputs
from my_torch.encoding.position import FenError
from my_torch.nn_module import losses
from my_torch.nn_module.layer import Layer
from my_torch.nn_module.neural_network_module import Neural_network_module

TRAINING_LINES = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White",
]


@pytest.fixture
def loadfile(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([INPUT_SIZE, 16, len(BASIC)], epochs=300,
                          learning_rate=0.05).save(str(path))
    return path


def chessfile(tmp_path, lines):
    path = tmp_path / "boards.txt"
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def loss(network_path, lines):
    """Loss of a saved network on labelled lines, with the loss and the label
    set it trains with."""
    samples = read_samples(lines)
    network = Neural_network_module.load(str(network_path))
    x = encode_all(s.position for s in samples)
    y = for_outputs(network.layers[-1].output).one_hot(s.label for s in samples)
    return losses.get(network.loss).value(network.forward(x), y)


def test_save_writes_savefile_and_keeps_loadfile(tmp_path, loadfile):
    savefile = tmp_path / "out.nn"
    original = loadfile.read_bytes()
    args = Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES)),
                     str(savefile))
    assert train(args) == 0
    assert loadfile.read_bytes() == original
    assert savefile.exists()


def test_without_save_loadfile_is_overwritten(tmp_path, loadfile):
    original = loadfile.read_bytes()
    args = Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES)))
    assert train(args) == 0
    assert loadfile.read_bytes() != original


def test_training_lowers_the_loss(tmp_path):
    """No validation: with 3 boards, holding one out would let early stopping
    keep epoch 1 on the say of a single board."""
    path = tmp_path / "net.nn"
    Neural_network_module([INPUT_SIZE, 16, len(BASIC)], epochs=50, learning_rate=0.05,
                          validation_split=0.0).save(str(path))
    before = loss(path, TRAINING_LINES)
    train(Arguments("train", str(path), str(chessfile(tmp_path, TRAINING_LINES))))
    assert loss(path, TRAINING_LINES) < before


def test_hyperparameters_are_kept(tmp_path, loadfile):
    train(Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES))))
    network = Neural_network_module.load(str(loadfile))
    assert (network.epochs, network.learning_rate) == (300, 0.05)


@pytest.mark.parametrize("lines, error, message", [
    (TRAINING_LINES + ["4k3/8/8/8/8/8/8/4K3 w - -"], FenError,
     "line 4: missing the expected output"),
    ([], ValueError, "no board to train on"),
    (["4k3/8/8/8/8/8/8/4K3 w - - Chek"], ValueError, "unknown label 'Chek'"),
    (["4k3/8/8/8/8/8/8/4K3 x - - Check"], FenError, "line 1: side to move"),
])
def test_bad_chess_file_leaves_the_network_untouched(tmp_path, loadfile, lines, error, message):
    original = loadfile.read_bytes()
    with pytest.raises(error, match=message):
        train(Arguments("train", str(loadfile), str(chessfile(tmp_path, lines))))
    assert loadfile.read_bytes() == original


def coloured(tmp_path, **hyperparameters):
    path = tmp_path / "coloured.nn"
    Neural_network_module([INPUT_SIZE, 16, len(FULL)], learning_rate=0.05,
                          validation_split=0.0, **hyperparameters).save(str(path))
    return path


def test_a_coloured_network_lowers_its_loss(tmp_path):
    path = coloured(tmp_path, epochs=50)
    before = loss(path, TRAINING_LINES)
    train(Arguments("train", str(path), str(chessfile(tmp_path, TRAINING_LINES))))
    assert loss(path, TRAINING_LINES) < before


def test_a_coloured_network_learns_the_colours(tmp_path):
    """Long enough on 3 boards to learn them by heart, colours included."""
    path = coloured(tmp_path, epochs=300)
    train(Arguments("train", str(path), str(chessfile(tmp_path, TRAINING_LINES))))
    samples = read_samples(TRAINING_LINES)
    scores = Neural_network_module.load(str(path)).forward(
        encode_all(s.position for s in samples))
    assert [FULL.label(i) for i in scores.argmax(axis=1)] == [s.label for s in samples]


def test_a_coloured_network_needs_coloured_labels(tmp_path):
    path = coloured(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="unknown label 'Check'"):
        train(Arguments("train", str(path),
                        str(chessfile(tmp_path, ["4k3/8/8/8/8/8/8/4K3 w - - Check"]))))
    assert path.read_bytes() == original


def test_network_that_does_not_fit_is_rejected(tmp_path):
    path = tmp_path / "net.nn"
    Neural_network_module([768, 8, 3]).save(str(path))
    with pytest.raises(ValueError, match="expected 769"):
        train(Arguments("train", str(path), str(chessfile(tmp_path, TRAINING_LINES))))


def test_missing_chess_file(tmp_path, loadfile):
    with pytest.raises(OSError):
        train(Arguments("train", str(loadfile), str(tmp_path / "missing.txt")))


def test_no_temporary_file_is_left(tmp_path, loadfile):
    train(Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES))))
    assert not list(tmp_path.glob("*.tmp"))


def test_a_crash_while_saving_keeps_loadfile(tmp_path, loadfile, monkeypatch):
    """Simulate a save that dies halfway through writing."""
    def broken_save(self, path):
        with open(path, "w") as file:
            file.write('{"version": 1, "data": {"lay')
        raise OSError("disk full")

    original = loadfile.read_bytes()
    monkeypatch.setattr(Neural_network_module, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        train(Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES))))
    assert loadfile.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_savefile_in_a_missing_directory(tmp_path, loadfile):
    savefile = tmp_path / "missing" / "out.nn"
    with pytest.raises(OSError):
        train(Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES)),
                        str(savefile)))
    assert not savefile.parent.exists()


@pytest.fixture
def always_nothing(tmp_path):
    """A network that answers Nothing to every board, and that a learning rate
    of 0 keeps that way: its accuracy is known in advance. No validation split,
    so the report is about all the boards."""
    layer = Layer(INPUT_SIZE, len(BASIC), "linear",
                  w=np.zeros((INPUT_SIZE, len(BASIC))), b=[1.0, 0.0, 0.0])
    path = tmp_path / "nothing.nn"
    Neural_network_module(layers=[layer], epochs=1, learning_rate=0.0,
                          validation_split=0.0).save(str(path))
    return path


def test_report_gives_the_boards_and_the_accuracy(tmp_path, always_nothing, capsys):
    """1 board out of 3 is Nothing."""
    train(Arguments("train", str(always_nothing), str(chessfile(tmp_path, TRAINING_LINES))))
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Number of boards : 3, 33.3% accuracy on the training boards (no validation)")


def test_no_report_when_the_save_fails(tmp_path, always_nothing, capsys):
    savefile = tmp_path / "missing" / "out.nn"
    with pytest.raises(OSError):
        train(Arguments("train", str(always_nothing),
                        str(chessfile(tmp_path, TRAINING_LINES)), str(savefile)))
    assert "Number of boards" not in capsys.readouterr().out


def test_no_report_for_a_bad_chess_file(tmp_path, always_nothing, capsys):
    with pytest.raises(ValueError):
        train(Arguments("train", str(always_nothing), str(chessfile(tmp_path, []))))
    assert capsys.readouterr().out == ""


EPOCH_LINE = re.compile(r"epoch (\d+)/(\d+): loss \d+\.\d{4}, validation loss (\d+\.\d{4}), "
                        r"validation accuracy (\d+\.\d)%")


def test_one_progress_line_per_epoch_then_the_report(tmp_path, loadfile, capsys):
    train(Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES))))
    *epochs, report = capsys.readouterr().out.splitlines()
    matches = [EPOCH_LINE.fullmatch(line) for line in epochs]
    assert all(matches)
    assert [int(m[1]) for m in matches] == list(range(1, len(epochs) + 1))
    assert {m[2] for m in matches} == {"300"}
    assert report.startswith("Number of boards : 3, kept epoch ")


def test_without_validation_the_progress_line_has_only_the_loss(tmp_path, always_nothing,
                                                                capsys):
    """Every board gets the scores [1, 0, 0], so softmax gives Nothing e / (e + 2)
    and the others 1 / (e + 2): the cross-entropy over the 3 boards is
    (ln((e + 2) / e) + 2 ln(e + 2)) / 3 = ln(e + 2) - 1 / 3."""
    train(Arguments("train", str(always_nothing), str(chessfile(tmp_path, TRAINING_LINES))))
    assert capsys.readouterr().out.splitlines()[0] == "epoch 1/1: loss 1.2181"


@pytest.fixture
def frozen_with_validation(tmp_path):
    """always_nothing with a validation split: its validation loss never
    changes, so it never improves after epoch 1 and patience 2 stops it at 3."""
    layer = Layer(INPUT_SIZE, len(BASIC), "linear",
                  w=np.zeros((INPUT_SIZE, len(BASIC))), b=[1.0, 0.0, 0.0])
    path = tmp_path / "frozen.nn"
    Neural_network_module(layers=[layer], epochs=10, learning_rate=0.0,
                          patience=2).save(str(path))
    return path


def test_the_report_names_the_kept_epoch_and_its_validation_accuracy(
        tmp_path, frozen_with_validation, capsys):
    train(Arguments("train", str(frozen_with_validation),
                    str(chessfile(tmp_path, TRAINING_LINES))))
    *epochs, report = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in epochs] == ["epoch 1/10", "epoch 2/10", "epoch 3/10"]
    first_accuracy = EPOCH_LINE.fullmatch(epochs[0])[4]
    assert report == (f"Number of boards : 3, kept epoch 1 of 3, "
                      f"{first_accuracy}% validation accuracy")


def test_the_kept_epoch_is_the_one_with_the_lowest_validation_loss(tmp_path, capsys):
    """Read back from the progress lines themselves, on a network that learns.
    The printed losses are rounded, so two epochs may print the same one: the
    kept epoch only has to print the lowest."""
    path = tmp_path / "net.nn"
    Neural_network_module([INPUT_SIZE, 16, len(BASIC)], epochs=15, learning_rate=0.05,
                          batch_size=4, patience=3).save(str(path))
    train(Arguments("train", str(path), str(chessfile(tmp_path, TRAINING_LINES * 10))))
    *epochs, report = capsys.readouterr().out.splitlines()
    matches = [EPOCH_LINE.fullmatch(line) for line in epochs]
    kept = re.fullmatch(r"Number of boards : 30, kept epoch (\d+) of (\d+), "
                        r"(\d+\.\d)% validation accuracy", report)
    assert int(kept[2]) == len(epochs)
    kept_line = matches[int(kept[1]) - 1]
    assert float(kept_line[3]) == min(float(m[3]) for m in matches)
    assert kept[3] == kept_line[4]
