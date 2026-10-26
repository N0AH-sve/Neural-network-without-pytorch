# my_torch

Neural network module written from scratch (no pytorch / tensorflow) and a
chessboard analyzer that predicts the state of a game given a FEN position:
`Checkmate`, `Check` or `Nothing`.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Build

```sh
make
```

This syncs the virtual environment with `uv` and produces the two binaries at
the root of the repository: `my_torch_analyzer` and `my_torch_generator`.

## Usage

```sh
./my_torch_analyzer --help
./my_torch_generator --help
```

## Makefile rules

| Rule | Description |
| --- | --- |
| `all` | build the binaries (default) |
| `install` | sync the uv environment |
| `run` | run the analyzer with `--help` |
| `tests_run` | run the test suite |
| `clean` | remove caches |
| `fclean` | `clean` + remove the binaries and the venv |
| `re` | `fclean` + `all` |

## Documentation

See [docs/](docs/).
