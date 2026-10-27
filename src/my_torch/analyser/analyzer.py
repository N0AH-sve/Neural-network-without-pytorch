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
