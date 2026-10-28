"""Entry point of the my_torch_analyzer binary."""

import sys

USAGE = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
    --train     Launch the neural network in training mode. Each chessboard in
                FILE must contain inputs to send to the neural network in FEN
                notation, and the expected output separated by space. If
                specified, the newly trained neural network will be saved in
                SAVEFILE. Otherwise, it will be saved in the original LOADFILE.
    --predict   Launch the neural network in prediction mode. Each chessboard in
                FILE must contain inputs to send to the neural network in FEN
                notation, and optionally an expected output.
    --save      Save neural network into SAVEFILE. Only works in train mode.
    LOADFILE    File containing an artificial neural network
    CHESSFILE   File containing chessboards"""

OPTIONS = ("--predict", "--train", "--save")


class AnalyzerError(ValueError):
    """Raised when the command line is not a valid analyzer invocation."""


class Analyzer:
    def __init__(self, argv: list[str]):
        self.loadfile, self.chessfile = self.parse_args(argv)

    @staticmethod
    def parse_args(argv: list[str]) -> tuple[str, str]:
        """Read '--predict LOADFILE CHESSFILE'; the flag may sit anywhere.

        Separating options from operands by their leading '-' keeps the two
        files in the order the user gave them, whatever the flag's position.
        """
        options = [arg for arg in argv if arg.startswith("-")]
        operands = [arg for arg in argv if not arg.startswith("-")]
        for option in options:
            if option not in OPTIONS:
                raise AnalyzerError(f"unknown option '{option}'")
        if "--train" in options:
            raise AnalyzerError("--train is not supported yet")
        if "--save" in options:
            raise AnalyzerError("--save only works in train mode")
        if "--predict" not in options:
            raise AnalyzerError("missing --predict")
        if options.count("--predict") > 1:
            raise AnalyzerError("--predict given more than once")
        if len(operands) != 2:
            raise AnalyzerError("invalid number of arguments")
        return operands[0], operands[1]


def run(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(USAGE)
        return 0

    print(USAGE, file=sys.stderr)
    return 84


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    try:
        return run(argv)
    except (OSError, ValueError) as error:
        print(f"my_torch_analyzer: {error}", file=sys.stderr)
        return 84


if __name__ == "__main__":
    sys.exit(main())
