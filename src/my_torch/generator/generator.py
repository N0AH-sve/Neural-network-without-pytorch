"""Entry point of the my_torch_generator binary."""

import json
import os
import sys

from my_torch.nn_module import activations
from my_torch.nn_module.neural_network_module import Neural_network_module

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i   Configuration file containing description of a neural
                    network we want to generate.
    nb_i            Number of neural networks to generate based on the
                    configuration file."""

INIT_STRATEGIES = ("normal",)
DEFAULT_INIT = "normal"


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
            if not nb.isdecimal() or int(nb) <= 0:
                raise GeneratorError(f"invalid number of networks: '{nb}'")
            jobs.append((path, int(nb)))
        return jobs

    def get_conf(self, path: str) -> dict:
        """Read a JSON configuration file. Its content is validated by check_conf."""
        with open(path) as file:
            try:
                return json.load(file)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise GeneratorError(
                    f"{os.path.basename(path)}: invalid JSON: {error}"
                ) from error

    @staticmethod
    def _check_layers(conf: dict, where: str) -> list[int]:
        layers = conf["layers"]
        if not isinstance(layers, list) or len(layers) < 2:
            raise GeneratorError(f"{where}: 'layers' must list at least 2 sizes")
        for i, size in enumerate(layers):
            if isinstance(size, bool) or not isinstance(size, int):
                raise GeneratorError(f"{where}: layer size {i} is not an integer")
            if size <= 0:
                raise GeneratorError(f"{where}: layer size {i} must be positive")
        return layers

    @staticmethod
    def _check_activations(conf: dict, layers: list[int], where: str) -> None:
        names = conf["activations"]
        if not isinstance(names, list):
            raise GeneratorError(f"{where}: 'activations' must be a list")
        if len(names) != len(layers) - 1:
            raise GeneratorError(
                f"{where}: {len(names)} activations for {len(layers)} layer sizes, "
                f"expected {len(layers) - 1} (one per connection)"
            )
        for i, name in enumerate(names):
            try:
                activations.get(name)
            except ValueError as error:
                raise GeneratorError(f"{where}: activation {i}: {error}") from error

    @staticmethod
    def _check_hyperparameters(conf: dict, where: str) -> None:
        hyper = conf.get("hyperparameters", {})
        if not isinstance(hyper, dict):
            raise GeneratorError(f"{where}: 'hyperparameters' must be an object")
        for key in ("epochs", "seed", "batch_size"):
            if key in hyper and (isinstance(hyper[key], bool)
                                 or not isinstance(hyper[key], int)):
                raise GeneratorError(f"{where}: '{key}' must be an integer")
        for key in ("epochs", "batch_size"):
            if key in hyper and hyper[key] <= 0:
                raise GeneratorError(f"{where}: '{key}' must be positive")
        if "learning_rate" in hyper:
            rate = hyper["learning_rate"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise GeneratorError(f"{where}: 'learning_rate' must be a number")
            if rate <= 0:
                raise GeneratorError(f"{where}: 'learning_rate' must be positive")

    def check_conf(self, conf: dict, path: str) -> None:
        """Reject a configuration before any network is written to disk."""
        where = os.path.basename(path)
        if not isinstance(conf, dict):
            raise GeneratorError(f"{where}: configuration must be an object")
        for key in ("layers", "activations"):
            if key not in conf:
                raise GeneratorError(f"{where}: missing key '{key}'")
        layers = self._check_layers(conf, where)
        self._check_activations(conf, layers, where)
        self._check_hyperparameters(conf, where)
        init = conf.get("init", DEFAULT_INIT)
        if init not in INIT_STRATEGIES:
            valid = ", ".join(sorted(INIT_STRATEGIES))
            raise GeneratorError(
                f"{where}: unknown init {init!r}, expected one of: {valid}"
            )

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
    if not argv:
        print(USAGE, file=sys.stderr)
        return 84
    try:
        Generator(argv).run()
    except (GeneratorError, OSError) as e:
        print(f"my_torch_generator: {e}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main())