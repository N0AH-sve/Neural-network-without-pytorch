# Architecture

```
.
├── Makefile                      build rules (all, install, run, tests_run, clean, fclean, re)
├── pyproject.toml                project metadata and dependencies (uv)
├── chess_files/                  example chess files, valid and broken, for the analyzer tests
├── config_files/                 example .conf files, valid and broken, for the generator tests
├── dataset/                      training boards by class, not versioned; mixed/ = shuffled split
├── loading_files/                example .nn files, valid and broken, for the save / load tests
├── src/my_torch/
│   ├── analyser/                 my_torch_analyzer
│   │   ├── analyzer.py           entry point: --help, then dispatch on the mode
│   │   ├── arguments.py          command line → Arguments (mode, files)
│   │   ├── network.py            load a .nn file and check it fits the encoding
│   │   ├── predict.py            --predict
│   │   └── train.py              --train
│   ├── generator/generator.py    entry point of my_torch_generator
│   ├── nn_module/                generic neural network module
│   │   ├── activations.py        linear, sigmoid, tanh, relu and their derivatives
│   │   ├── layer.py              dense layer: forward, backward, update
│   │   └── neural_network_module.py  network: training, save / load (JSON)
│   ├── encoding/                 chess-specific input / output encoding
│   │   ├── position.py           FEN string → Position (parsing and validation)
│   │   ├── chessfile.py          chess file line → Sample (position + label)
│   │   ├── features.py           Position → 769-value input vector
│   │   └── labels.py             label ⇄ output neuron, one-hot targets
│   └── tools/
│       └── shuffle_dataset.py    mix, deduplicate, shuffle and split chess files
├── tests/                        test suite (pytest), one file per module
└── docs/                         this documentation
```

`nn_module` knows nothing about chess, as the subject asks for a generic
module; everything chess-specific lives in `encoding`.

## Data flow

**Predict mode** — labels in the file, if any, are ignored:

| Step | Code | Result | Status |
| --- | --- | --- | --- |
| 1. Load the network | `Neural_network_module.load(loadfile)`, input size checked | network with 769 inputs | done |
| 2. Pick the label set | `Predict.label_set_of`: 3 outputs → `BASIC`, 5 → `FULL` | `LabelSet` | done |
| 3. Read the file | `chessfile.read_samples(file)` | `list[Sample]` | done |
| 4. Encode the boards | `features.encode_all(s.position for s in samples)` | `X`, shape `(N, 769)` | done |
| 5. Run the network | `network.forward(X)` | scores, shape `(N, classes)` | done |
| 6. Pick each answer | `argmax` per row, then `labels.label(i)` | one label per line on stdout | done |

All of it runs in `analyser/predict.py`. The whole file is read before the
first prediction is printed, so a broken board never leaves a truncated list.

**Train mode** — every line must have a label:

| Step | Code | Result | Status |
| --- | --- | --- | --- |
| 1. Load the network | `analyser.network.load_network(loadfile)` | network with 769 inputs, 3 outputs | done |
| 2. Read the file | `chessfile.read_samples(file, require_labels=True)` | `list[Sample]` | done |
| 3. Encode the boards | `features.encode_all(s.position for s in samples)` | `X`, shape `(N, 769)`, booleans | done |
| 4. Encode the labels | `BASIC.one_hot(s.label for s in samples)` | `Y`, shape `(N, 3)` | done |
| 5. Hold out validation | `validation_split` of the boards, drawn once from `seed`, inside `network.train` | training and validation indices | done |
| 6. Run the network | `network.forward(X[batch])`, inside `network.train`, on shuffled batches of `batch_size` boards | scores, shape `(batch_size, 3)` | done |
| 7. Compare with `Y` | loss on scores and `Y[batch]` | gradient of the loss | mean squared error (`_mean_squared_error`); softmax + cross-entropy to do |
| 8. Learn | `network.backward(gradient)`, `network.update(learning_rate)`, after every batch | new weights and biases | done |
| 9. Validate and maybe stop | after every epoch: validation loss and accuracy; stop after `patience` epochs without improvement, then restore the best epoch's weights | `TrainingHistory`, with `best` the kept epoch | done |
| 10. Save | written to `<path>.tmp`, then `os.replace` | `.nn` file (JSON) in `SAVEFILE` or `LOADFILE` | done |
| 11. Report | one line per epoch while training (`on_epoch`), then the kept epoch and its validation accuracy | stdout | done |

All of it runs in `analyser/train.py`. `Y` is never fed to the network: it is
only compared with the network's output in step 7.

## Input encoding

Each board becomes 769 booleans (`features.INPUT_SIZE`):

| Indices | Meaning |
| --- | --- |
| `plane * 64 + row * 8 + col` | `True` if the piece of `plane` is on that square |
| `768` (last) | `True` if White is to move |

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
- **Booleans, not floats**: 1 byte per value instead of 8, so the 572k
  training boards take 440 MB instead of 3.5 GB. numpy turns each batch into
  floats when it meets the weights, and `--train` measures its accuracy with
  `forward_in_batches`, so training never makes a full float copy. (`--predict`
  still runs `forward` on the whole file, which is fine for small files.)
- **Side to move** is needed for the coloured labels: in a legal position only
  the side to move can be in check.
- **Castling, en passant and the move counters are left out**: they are
  validated but do not change whether a king is in check or mated.

## Output classes

The classes are a `labels.LabelSet`, in the order of the output neurons:
`BASIC` (`Nothing`, `Check`, `Checkmate`) or `FULL` (with the colour of the side
giving check or mate). See [formats.md](formats.md#label).

The network only outputs one score per class and knows nothing about their
meaning: the analyzer translates them. `--train` uses `BASIC`, the set the
subject requires, and refuses a network without exactly 3 outputs; `--predict`
picks `BASIC` or `FULL` from the number of outputs (3 or 5).

## Planned modules

- `nn_module/losses.py` — MSE, softmax + cross entropy.
- One loading rule for both modes: `--predict` checks the network itself and
  accepts 3 or 5 outputs, `--train` uses `analyser.network.load_network` and
  accepts 3 only.

## Design notes

To be filled during the implementation: justification of the hyperparameters
(number of layers, neurons, learning rate, batch size, initialization strategy)
is required at the defense.
