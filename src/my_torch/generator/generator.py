"""Entry point of the my_torch_generator binary."""

import sys
import os

from my_torch.nn_module.neural_network_module import Neural_network_module

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i   Configuration file containing description of a neural
                    network we want to generate.
    nb_i            Number of neural networks to generate based on the
                    configuration file."""


class GeneratorError(Exception):
    pass


class Generator:
    def __init__(self, argv: list[str]):
        self.jobs: list[tuple[str, int]] = self.parse_args(argv)

    @staticmethod
    def parse_args(argv: list[str]) -> list[tuple[str, int]]:
        if not argv or len(argv) % 2 != 0:
            raise GeneratorError("invalid number of arguments")
        jobs = []
        for path, nb in zip(argv[0::2], argv[1::2]):
            if not nb.isdigit() or int(nb) <= 0:
                raise GeneratorError(f"invalid number of networks: '{nb}'")
            jobs.append((path, int(nb)))
        return jobs

    def get_conf(self, path: str) -> dict:
        pass
    def check_conf(self, conf: dict, path: str) -> None:
        pass

    def run(self) -> None:
        confs = []
        for path, nb in self.jobs:
            conf = self.get_conf(path)
            self.check_conf(conf, path)
            confs.append((path, conf, nb))
        for path, conf, nb in confs:
            base = os.path.splitext(os.path.basename(path))[0]
            for i in range(1, nb + 1):
                net = Neural_network_module.from_config(conf)
                net.save(f"{base}_{i}.nn")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        print(USAGE)
        return 0
    try:
        Generator(argv).run()
    except (GeneratorError, OSError) as e:
        print(f"my_torch_generator: {e}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main())