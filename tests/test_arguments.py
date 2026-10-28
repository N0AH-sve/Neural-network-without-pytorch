import pytest

from my_torch.analyser.arguments import Arguments, UsageError, parse_args


@pytest.mark.parametrize("argv, expected", [
    (["--train", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt")),
    (["--predict", "net.nn", "boards.txt"], Arguments("predict", "net.nn", "boards.txt")),
    (["--train", "--save", "out.nn", "net.nn", "boards.txt"],
     Arguments("train", "net.nn", "boards.txt", "out.nn")),
])
def test_valid_command_lines(argv, expected):
    assert parse_args(argv) == expected


@pytest.mark.parametrize("argv", [
    ["net.nn", "boards.txt", "--train", "--save", "out.nn"],
    ["--save", "out.nn", "--train", "net.nn", "boards.txt"],
    ["net.nn", "--train", "boards.txt", "--save", "out.nn"],
])
def test_options_may_come_in_any_order(argv):
    assert parse_args(argv) == Arguments("train", "net.nn", "boards.txt", "out.nn")


def test_files_keep_their_order():
    """The first file is always the network, the second the boards."""
    args = parse_args(["--predict", "boards.txt", "net.nn"])
    assert (args.loadfile, args.chessfile) == ("boards.txt", "net.nn")


@pytest.mark.parametrize("argv, message", [
    (["net.nn", "boards.txt"], "missing --train or --predict"),
    (["--train", "--predict", "net.nn", "boards.txt"], "exactly one of"),
    (["--train", "--train", "net.nn", "boards.txt"], "exactly one of"),
    (["--predict", "--save", "out.nn", "net.nn", "boards.txt"], "only works with --train"),
    (["--train", "net.nn", "boards.txt", "--save"], "needs a SAVEFILE"),
    (["--train", "--save", "a.nn", "--save", "b.nn", "net.nn", "boards.txt"], "given twice"),
    (["--train", "net.nn"], "got 1 file"),
    (["--train"], "got 0 file"),
    (["--train", "net.nn", "boards.txt", "extra.txt"], "got 3 file"),
    (["--train", "-v", "net.nn", "boards.txt"], "unknown option '-v'"),
    (["--trian", "net.nn", "boards.txt"], "unknown option '--trian'"),
    ([], "missing --train or --predict"),
])
def test_invalid_command_lines(argv, message):
    with pytest.raises(UsageError, match=message):
        parse_args(argv)


def test_usage_error_is_a_value_error():
    """main() turns every ValueError into exit 84."""
    assert issubclass(UsageError, ValueError)
