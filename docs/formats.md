# File formats

## Network file (`.nn`)

The format is free. It must be named with the `my_torch_network` prefix for the
pre-trained network pushed for the autograder.

Information to store: layer sizes, activation per layer, weights, biases, and
the hyperparameters used for training.

## Configuration file (`.conf`)

Input of `my_torch_generator`: description of the topology of a network to
generate (layer sizes, activations, initialization strategy).

## Hyperparameters

Both files carry the training hyperparameters in a `"hyperparameters"` object;
the generator copies them from the `.conf` into every `.nn` it writes. Any of
them may be left out, which gives the default:

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | `20` | passes over the training boards |
| `learning_rate` | `0.005` | size of each gradient step |
| `batch_size` | `64` | boards per weight update; at least the number of boards gives full-batch descent |
| `seed` | `42` | order in which the boards are shuffled at each epoch |

`epochs` and `batch_size` must be positive integers, `learning_rate` a positive
number, `seed` any integer. `.nn` files written before `batch_size` existed load
with the default.

## Chessboard file

One board per line, in
[Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation),
optionally followed by its label. Parsed by `my_torch.encoding.chessfile`.

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black
```

| FEN: board | FEN: side | FEN: castling | FEN: en passant | FEN: halfmove | FEN: fullmove | Label |
| --- | --- | --- | --- | --- | --- | --- |
| `rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR` | `w` | `KQkq` | `-` | `1` | `3` | `Checkmate Black` |

### FEN

A FEN contains spaces itself: it has 6 fields, the last 2 being optional.

| Field | Example | Accepted values |
| --- | --- | --- |
| board | `rnb1kbnr/.../RNBQKBNR` | 8 ranks separated by `/`, from rank 8 to rank 1; pieces `PNBRQK` (white) and `pnbrqk` (black), digits `1`–`8` for empty squares; each rank exactly 8 squares; exactly one `K` and one `k` |
| side to move | `w` | `w` or `b` |
| castling | `KQkq` | `-` or a subset of `KQkq`, in that order |
| en passant | `-` | `-` or a square on rank 3 or 6, e.g. `d3` |
| halfmove clock | `1` | non-negative integer, default `0` when both counters are omitted |
| fullmove number | `3` | non-negative integer, default `1` when both counters are omitted |

Only the syntax and the king count are checked: a readable but illegal board
(adjacent kings, pawn on rank 1…) is accepted and left to the network.

### Label

Everything after the FEN is the label: it starts at the first word beginning
with a letter, so it is never mistaken for a move counter.

- **Train mode:** the label is required.
- **Predict mode:** the label is optional and ignored for the prediction.

Labels ignore case and extra spaces. The accepted values depend on the classes
of the network:

| Label set | Classes |
| --- | --- |
| `BASIC` | `Nothing`, `Check`, `Checkmate` |
| `FULL` | `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black` |

A network without colours (`BASIC`) also accepts coloured labels and ignores
the colour, so the same file can train both kinds of networks.

**The colour names the side giving check or mate**, not the side under attack.
Both examples come from the subject:

| Board | Situation | Label |
| --- | --- | --- |
| `rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3` | Fool's mate: White to move and mated | `Checkmate Black` |
| `rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3` | White queen on h5 checks the black king | `Check White` |

### Robustness

Blank lines are skipped; Windows line endings, tabs and repeated spaces are
accepted. An invalid line is reported with its number, e.g.
`line 3: side to move: expected 'w' or 'b', got 'W'`.
