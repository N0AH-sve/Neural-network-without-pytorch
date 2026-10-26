# File formats

## Network file (`.nn`)

The format is free. It must be named with the `my_torch_network` prefix for the
pre-trained network pushed for the autograder.

Information to store: layer sizes, activation per layer, weights, biases, and
the hyperparameters used for training.

## Configuration file (`.conf`)

Input of `my_torch_generator`: description of the topology of a network to
generate (layer sizes, activations, initialization strategy).

## Chessboard file

One board per line, in
[Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).
In train mode, the expected output (`Checkmate`, `Check` or `Nothing`) follows
the FEN, separated by a space.
