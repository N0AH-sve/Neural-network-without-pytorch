# Architecture

```
.
├── Makefile              build rules (all, install, run, tests_run, clean, fclean, re)
├── pyproject.toml        project metadata and dependencies (uv)
├── src/my_torch/
│   ├── __init__.py
│   ├── analyzer.py       entry point of my_torch_analyzer
│   └── generator.py      entry point of my_torch_generator
├── tests/                test suite (pytest)
└── docs/                 this documentation
```

## Planned modules

- `layer.py` — dense layer, forward and backward pass.
- `network.py` — network container, save / load, training loop.
- `activations.py` — sigmoid, tanh, relu, softmax and their derivatives.
- `losses.py` — MSE, cross entropy.
- `encoding.py` — FEN string to input vector, output vector to label.

## Design notes

To be filled during the implementation: justification of the hyperparameters
(number of layers, neurons, learning rate, batch size, initialization strategy)
is required at the defense.
