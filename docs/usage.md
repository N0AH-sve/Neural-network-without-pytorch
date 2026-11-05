# Usage

## my_torch_analyzer

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

- `--train` — train the network on `CHESSFILE`; each line holds a FEN board and
  the expected output separated by a space. The trained network is saved in
  `SAVEFILE` if given, otherwise in `LOADFILE`.
- `--predict` — print the prediction of every board of `CHESSFILE`, in order.
- `--save SAVEFILE` — only valid in train mode.

Options may come before or after the two files; the first file is always
`LOADFILE`, the second `CHESSFILE`. Line format: see
[formats.md](formats.md#chessboard-file).

### Train mode

```sh
$> ./my_torch_analyzer --train --save trained.nn my_torch_network.nn boards.txt
epoch 1/30: loss 1.0726, validation loss 1.0625, validation accuracy 42.0%
...
epoch 13/30: loss 0.8054, validation loss 0.9188, validation accuracy 56.0%
epoch 14/30: loss 0.7846, validation loss 0.9215, validation accuracy 55.7%
epoch 15/30: loss 0.7636, validation loss 0.9243, validation accuracy 56.1%
epoch 16/30: loss 0.7430, validation loss 0.9289, validation accuracy 56.2%
Number of boards : 20000, kept epoch 13 of 16, 56.0% validation accuracy
```

1. `LOADFILE` is loaded and must take 769 inputs (one encoded board) and give
   3 outputs (`Nothing`, `Check`, `Checkmate`).
2. Every line of `CHESSFILE` must have its expected output.
3. `validation_split` of the boards (20% by default) is kept aside and never
   trained on.
4. The network is trained on the other boards with the hyperparameters stored
   in `LOADFILE` (see [formats.md](formats.md#hyperparameters)): at each epoch
   the boards are shuffled, then the weights are updated after every
   `batch_size` boards. One line per epoch gives the training loss and the
   validation loss and accuracy.
5. Training stops early once the validation loss has not improved for
   `patience` epochs, and the weights of the best epoch are kept: the epochs
   after it only learned the training boards by heart. This is how training
   avoids overfitting.
6. It is saved in `SAVEFILE`, or back in `LOADFILE`. The file is written to
   `<path>.tmp` then renamed, so a crash never leaves a half-written network.
7. Once the save succeeded, the last line gives the kept epoch and its
   validation accuracy: how the network does on boards it never learned from.
   With `validation_split` set to `0`, it gives the accuracy on the training
   boards instead, marked `(no validation)`.

Everything random follows `seed`, so the same `LOADFILE` trained on the same
file always gives the same network.

On any error, nothing is saved and `LOADFILE` is left untouched.

The wrapper uses 4 OpenBLAS threads unless `OPENBLAS_NUM_THREADS` is set, which
is the fastest on small batches (see
[benchmarks.md](benchmarks.md#training-speed-batch-size-and-openblas-threads)):

```sh
$> OPENBLAS_NUM_THREADS=8 ./my_torch_analyzer --train my_torch_network.nn boards.txt
```

| Error | Example message |
| --- | --- |
| wrong command line | `expected LOADFILE and CHESSFILE, got 1 file(s)` |
| missing file | `[Errno 2] No such file or directory: 'net.nn'` |
| network of the wrong size | `net.nn: network takes 768 inputs, expected 769` |
| invalid board | `line 3: side to move: expected 'w' or 'b', got 'W'` |
| board without expected output | `line 2: missing the expected output` |
| unknown expected output | `unknown label 'Chek', expected one of: Nothing, Check, Checkmate` |
| empty file | `boards.txt: no board to train on` |

## my_torch_generator

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

Generates `nb_i` random networks from `config_file_i`, named
`<config_name>_<i>.nn`.

## Mixing the dataset

The boards in `dataset/` are sorted by class (`check/`, `checkmate/`,
`nothing/`), but `--train` takes a single file, and training on one class at a
time teaches the network nothing. This tool merges them into a shuffled
training file and a validation file:

```sh
$> uv run python -m my_torch.tools.shuffle_dataset
715555 boards read, 7464 duplicates removed
dataset/mixed/train.txt: 572444 boards (238129 Nothing, 198484 Check, 135831 Checkmate)
dataset/mixed/validation.txt: 143111 boards (59434 Nothing, 49649 Check, 34028 Checkmate)
$> ./my_torch_analyzer --train my_torch_network.nn dataset/mixed/train.txt
```

| Option | Default | Meaning |
| --- | --- | --- |
| `INPUT ...` | `dataset` | chess files, or folders whose `.txt` files are read |
| `-o OUTDIR` | `dataset/mixed` | folder that receives `train.txt` and `validation.txt` |
| `--seed N` | `42` | shuffle seed |
| `--validation RATIO` | `0.2` | share of the boards kept for validation, in `[0, 1)` |
| `--limit N` | all | keep only `N` boards picked at random, for quick experiments |

- **Input files are only read**: only `train.txt` and `validation.txt` are
  written.
- **Every line is parsed** like `--train` does, so a broken line is reported
  with its file and line number before any training starts.
- **Duplicates are removed**: a board (placement and side to move) is kept
  once, so the same position cannot be in both files and make the validation
  score look better than it is.
- **The split is reproducible**: the same inputs and seed always give the same
  files. Earlier outputs (`-o` and `dataset/mixed`) are never read back as
  inputs.
- Errors exit with `84`; invalid option values are reported by `argparse`.

`dataset/` is not versioned (see `.gitignore`): keep a copy of the boards, as
they must be shown at the defense.

## Error handling

Errors are printed on the error output and the program exits with `84`.
A successful run exits with `0`.
