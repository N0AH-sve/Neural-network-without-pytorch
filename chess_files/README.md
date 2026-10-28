# Chess files

Test fixtures for `./my_torch_analyzer --predict LOADFILE CHESSFILE`.

A line holds a FEN board (4 or 6 fields) optionally followed by its expected
state. In prediction mode the state is ignored; it is there for train mode.

## Label convention

A colour names the side **giving** check or mate, not the side under attack.
So `rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3` is White to
move and mated, hence `Checkmate Black`. This matches the subject's own output.

## Provenance

Positions come from random self-play games; every label was computed with
`python-chess` 1.11.2, which is **not** a dependency of this project: it was
installed in a throwaway virtualenv to label the data, nothing imports it.

## Valid, unlabelled

| File | Lines | Content |
|---|---|---|
| `subject.txt` | 5 | the five boards printed in the subject |
| `unlabelled.txt` | 60 | 20 of each state, shuffled, no label |

A correct network on `subject.txt` outputs `Checkmate`, `Nothing`, `Nothing`,
`Check`, `Nothing` (or with colours: `Checkmate Black`, `Nothing`, `Nothing`,
`Check White`, `Nothing`). The subject's own `basic` network gets the first one
wrong, so its listing is a prediction, not the ground truth.

## Valid, labelled

| File | Lines | Content |
|---|---|---|
| `labelled.txt` | 60 | the `unlabelled.txt` boards, 3-class labels, 20/20/20 |
| `labelled_coloured.txt` | 60 | the same boards, 5-class labels |
| `nothing.txt` | 20 | one class only |
| `check.txt` | 20 | one class only |
| `checkmate.txt` | 20 | one class only |

## Valid, format edge cases

| File | Expects |
|---|---|
| `four_fields.txt` | FEN without the move counters: 4 fields are enough |
| `blank_lines.txt` | blank lines skipped, 2 predictions for 6 lines |
| `trailing_spaces.txt` | leading and trailing spaces ignored |
| `empty.txt` | no board, no output, exit 0 |
| `unknown_label.txt` | `Draw` ignored in predict mode; train must reject it |

## Invalid: every one must exit 84 with nothing on stdout

| File | Error |
|---|---|
| `invalid_one_field.txt` | `line 2: expected 4 or 6 fields, got 1` |
| `invalid_five_fields.txt` | `expected 4 or 6 fields, got 5` |
| `invalid_rank_too_long.txt` | `rank 7 has 9 squares, expected 8` |
| `invalid_rank_too_short.txt` | `rank 7 has 7 squares, expected 8` |
| `invalid_rank_count.txt` | `expected 8 ranks, got 7` |
| `invalid_char.txt` | `invalid character 'x' in rank 8` |
| `invalid_side.txt` | `side to move: expected 'w' or 'b', got 'x'` |
| `invalid_castling.txt` | `castling: expected '-' or a subset of 'KQkq'` |
| `invalid_en_passant.txt` | `en passant: expected '-' or a square on rank 3 or 6` |
| `invalid_halfmove.txt` | `halfmove: expected a non-negative integer` |
| `missing_black_king.txt` | `expected exactly one 'k', got 0` |
| `two_white_kings.txt` | `expected exactly one 'K', got 2` |
| `error_on_last_line.txt` | 10 valid boards then a broken one: stdout stays **empty** |

`error_on_last_line.txt` is the one that matters most: it proves the whole file
is parsed before the first line is printed, so a late error cannot leave a
truncated prediction list behind.
