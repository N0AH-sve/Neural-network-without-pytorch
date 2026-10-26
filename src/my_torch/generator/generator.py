"""Entry point of the my_torch_generator binary."""

import sys

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i   Configuration file containing description of a neural
                    network we want to generate.
    nb_i            Number of neural networks to generate based on the
                    configuration file."""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if "--help" in argv or "-h" in argv:
        print(USAGE)
        return 0

    print(USAGE, file=sys.stderr)
    return 84


if __name__ == "__main__":
    sys.exit(main())
