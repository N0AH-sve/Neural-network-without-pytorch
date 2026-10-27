import pytest

from my_torch.generator import generator
from my_torch.analyser import analyzer


@pytest.mark.parametrize("module", [analyzer, generator])
def test_help_exits_zero(module, capsys):
    assert module.main(["--help"]) == 0
    assert capsys.readouterr().out.startswith("USAGE")


@pytest.mark.parametrize("module", [analyzer, generator])
def test_no_argument_exits_84(module, capsys):
    assert module.main([]) == 84
    assert capsys.readouterr().err.startswith("USAGE")


@pytest.mark.parametrize("error", [
    ValueError("layer 0: missing key 'b'"),
    FileNotFoundError("no such file"),
])
def test_analyzer_turns_a_load_error_into_84(monkeypatch, capsys, error):
    """load() raises ValueError or OSError; main() is where they become exit 84."""
    def failing_run(argv):
        raise error

    monkeypatch.setattr(analyzer, "run", failing_run)
    assert analyzer.main(["--predict", "net.nn", "boards.txt"]) == 84
    assert str(error) in capsys.readouterr().err


def test_analyzer_does_not_swallow_a_successful_run(monkeypatch):
    monkeypatch.setattr(analyzer, "run", lambda argv: 0)
    assert analyzer.main([]) == 0
