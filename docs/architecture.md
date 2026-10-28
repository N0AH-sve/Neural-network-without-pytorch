# Architecture

```
.
├── Makefile                      build rules (all, install, run, tests_run, clean, fclean, re)
├── pyproject.toml                project metadata and dependencies (uv)
├── src/my_torch/
│   ├── analyser/analyzer.py      entry point of my_torch_analyzer
│   ├── generator/generator.py    entry point of my_torch_generator
│   ├── nn_module/                generic neural network module
│   │   ├── activations.py        linear, sigmoid, tanh, relu and their derivatives
│   │   ├── layer.py              dense layer: forward, backward, update
│   │   └── neural_network_module.py  network: layers, training loop
│   └── encoding/                 chess-specific input / output encoding
│       ├── position.py           FEN string → Position (parsing and validation)
│       ├── chessfile.py          chess file line → Sample (position + label)
│       ├── features.py           Position → 769-value input vector
│       └── labels.py             label ⇄ output neuron, one-hot targets
├── tests/                        test suite (pytest), one file per module
└── docs/                         this documentation
```

`nn_module` knows nothing about chess, as the subject asks for a generic
module; everything chess-specific lives in `encoding`.

## Data flow

**Predict mode** — labels in the file, if any, are ignored:

| Step | Code | Result | Status |
| --- | --- | --- | --- |
| 1. Read the file | `chessfile.read_samples(file)` | `list[Sample]` | done |
| 2. Encode the boards | `features.encode_all(s.position for s in samples)` | `X`, shape `(N, 769)` | done |
| 3. Run the network | `network.forward(X)` | scores, shape `(N, classes)` | done |
| 4. Pick each answer | `argmax` per row, then `LabelSet.label(i)` | one label per line on stdout | to do |

**Train mode** — every line must have a label:

| Step | Code | Result | Status |
| --- | --- | --- | --- |
| 1. Read the file | `chessfile.read_samples(file)` | `list[Sample]` | done |
| 2. Encode the boards | `features.encode_all(s.position for s in samples)` | `X`, shape `(N, 769)` | done |
| 3. Encode the labels | `LabelSet.one_hot(s.label for s in samples)` | `Y`, shape `(N, classes)` | done |
| 4. Run the network | `network.forward(X)` | scores, shape `(N, classes)` | done |
| 5. Compare with `Y` | loss on scores and `Y` | gradient of the loss | MSE inline in `train`; `losses.py` to do |
| 6. Learn | `network.backward(gradient)`, `network.update(lr)` | new weights and biases | done |
| 7. Save | write the network | `.nn` file | to do |

`Y` is never fed to the network: it is only compared with the network's output
in step 5. The analyzer itself is not wired to these steps yet.

## Input encoding

Each board becomes 769 values (`features.INPUT_SIZE`):

| Indices | Meaning |
| --- | --- |
| `plane * 64 + row * 8 + col` | 1 if the piece of `plane` is on that square, else 0 |
| `768` (last) | 1 if White is to move, 0 if Black |

| Plane | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Piece | `P` | `N` | `B` | `R` | `Q` | `K` | `p` | `n` | `b` | `r` | `q` | `k` |

Row 0 is rank 8 (the FEN reading order) and column 0 is file a: `a8` is square
0, `h8` is 7, `a1` is 56, `e1` is 60, `h1` is 63. The white king on e1 is
therefore index `5 * 64 + 60 = 380`.

Why this layout:

- **One 0/1 value per (piece, square)** rather than one number per square with a
  piece code: a code would tell the network that a queen (5) is worth 5 pawns
  (1) through the same weight, an ordering that means nothing. Here every
  piece/square pair has its own weights.
- **Side to move** is needed for the coloured labels: in a legal position only
  the side to move can be in check.
- **Castling, en passant and the move counters are left out**: they are
  validated but do not change whether a king is in check or mated.

## Output classes

The classes are a `labels.LabelSet`, in the order of the output neurons:
`BASIC` (`Nothing`, `Check`, `Checkmate`) or `FULL` (with the colour of the side
giving check or mate). See [formats.md](formats.md#label).

## Planned modules

- `nn_module/losses.py` — MSE, softmax + cross entropy.
- Save / load of a network in the `.nn` format, storing its label set.

## Design notes

To be filled during the implementation: justification of the hyperparameters
(number of layers, neurons, learning rate, batch size, initialization strategy)
is required at the defense.
