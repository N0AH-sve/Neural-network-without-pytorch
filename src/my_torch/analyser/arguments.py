"""Command line of my_torch_analyzer."""

from dataclasses import dataclass
from typing import Literal


class UsageError(ValueError):
    """Raised when the command line does not follow the usage."""


@dataclass(frozen=True)
class Arguments:
    mode: Literal["train", "predict"]
    loadfile: str
    chessfile: str
    savefile: str | None = None


def parse_args(argv: list[str]) -> Arguments:
    """Parse [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE.

    Options may come in any order, before or after the files.
    """
    mode = None
    savefile = None
    files = []
    args = iter(argv)
    for arg in args:
        if arg in ("--train", "--predict"):
            if mode is not None:
                raise UsageError("give exactly one of --train and --predict")
            mode = arg.removeprefix("--")
        elif arg == "--save":
            if savefile is not None:
                raise UsageError("--save given twice")
            savefile = next(args, None)
            if savefile is None:
                raise UsageError("--save needs a SAVEFILE")
        elif arg.startswith("-"):
            raise UsageError(f"unknown option {arg!r}")
        else:
            files.append(arg)
    if mode is None:
        raise UsageError("missing --train or --predict")
    if savefile is not None and mode != "train":
        raise UsageError("--save only works with --train")
    if len(files) != 2:
        raise UsageError(f"expected LOADFILE and CHESSFILE, got {len(files)} file(s)")
    return Arguments(mode, *files, savefile)
