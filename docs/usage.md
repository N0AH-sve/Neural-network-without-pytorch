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

## my_torch_generator

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

Generates `nb_i` random networks from `config_file_i`, named
`<config_name>_<i>.nn`.

## Error handling

Errors are printed on the error output and the program exits with `84`.
A successful run exits with `0`.
