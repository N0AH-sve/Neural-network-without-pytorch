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
