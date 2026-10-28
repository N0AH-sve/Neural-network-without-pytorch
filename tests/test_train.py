import numpy as np
import pytest

from my_torch.analyser.arguments import Arguments
from my_torch.analyser.train import train
from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import INPUT_SIZE, encode_all
from my_torch.encoding.labels import BASIC
from my_torch.encoding.position import FenError
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
    """Mean squared error of a saved network on labelled lines."""
    samples = read_samples(lines)
    x = encode_all(s.position for s in samples)
    y = BASIC.one_hot(s.label for s in samples)
    output = Neural_network_module.load(str(network_path)).forward(x)
    return float(np.mean((output - y) ** 2))


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


def test_training_lowers_the_loss(tmp_path, loadfile):
    before = loss(loadfile, TRAINING_LINES)
    train(Arguments("train", str(loadfile), str(chessfile(tmp_path, TRAINING_LINES))))
    assert loss(loadfile, TRAINING_LINES) < before


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
    of 0 keeps that way: its accuracy is known in advance."""
    layer = Layer(INPUT_SIZE, len(BASIC), "linear",
                  w=np.zeros((INPUT_SIZE, len(BASIC))), b=[1.0, 0.0, 0.0])
    path = tmp_path / "nothing.nn"
    Neural_network_module(layers=[layer], epochs=1, learning_rate=0.0).save(str(path))
    return path


def test_report_gives_the_boards_and_the_accuracy(tmp_path, always_nothing, capsys):
    """1 board out of 3 is Nothing."""
    train(Arguments("train", str(always_nothing), str(chessfile(tmp_path, TRAINING_LINES))))
    assert capsys.readouterr().out == "Number of boards : 3, 33.3% accuracy\n"


def test_no_report_when_the_save_fails(tmp_path, always_nothing, capsys):
    savefile = tmp_path / "missing" / "out.nn"
    with pytest.raises(OSError):
        train(Arguments("train", str(always_nothing),
                        str(chessfile(tmp_path, TRAINING_LINES)), str(savefile)))
    assert capsys.readouterr().out == ""


def test_no_report_for_a_bad_chess_file(tmp_path, always_nothing, capsys):
    with pytest.raises(ValueError):
        train(Arguments("train", str(always_nothing), str(chessfile(tmp_path, []))))
    assert capsys.readouterr().out == ""
