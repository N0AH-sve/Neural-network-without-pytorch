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
Number of boards : 3, 100.0% accuracy
```

1. `LOADFILE` is loaded and must take 769 inputs (one encoded board) and give
   3 outputs (`Nothing`, `Check`, `Checkmate`).
2. Every line of `CHESSFILE` must have its expected output.
3. The network is trained with the epochs and learning rate stored in
   `LOADFILE`.
4. It is saved in `SAVEFILE`, or back in `LOADFILE`. The file is written to
   `<path>.tmp` then renamed, so a crash never leaves a half-written network.
5. The number of boards and the accuracy are printed once the save succeeded.
   This accuracy is measured on the training boards themselves: it shows the
   network learned them, not that it generalises.

On any error, nothing is saved and `LOADFILE` is left untouched.

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

## Error handling

Errors are printed on the error output and the program exits with `84`.
A successful run exits with `0`.
