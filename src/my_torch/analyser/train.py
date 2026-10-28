"""--train mode of my_torch_analyzer."""

import os

import numpy as np

from my_torch.analyser.arguments import Arguments
from my_torch.analyser.network import load_network
from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import encode_all
from my_torch.encoding.labels import BASIC
from my_torch.nn_module.neural_network_module import Neural_network_module


def train(args: Arguments) -> int:
    """Train LOADFILE on the labelled boards of CHESSFILE, then save it in
    SAVEFILE if given, otherwise back in LOADFILE."""
    network = load_network(args.loadfile)
    with open(args.chessfile) as file:
        samples = read_samples(file, require_labels=True)
    if not samples:
        raise ValueError(f"{args.chessfile}: no board to train on")
    x = encode_all(sample.position for sample in samples)
    y = BASIC.one_hot(sample.label for sample in samples)
    network.train(x, y)
    accuracy = np.mean(network.forward(x).argmax(axis=1) == y.argmax(axis=1))
    print(f"Number of boards : {len(samples)}, {accuracy:.1%} accuracy")
    _save(network, args.savefile or args.loadfile)
    return 0


def _save(network: Neural_network_module, path: str) -> None:
    """Write next to the target, then rename: os.replace is atomic, so a crash
    while writing never leaves a half-written file in place of LOADFILE."""
    temporary = f"{path}.tmp"
    try:
        network.save(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
