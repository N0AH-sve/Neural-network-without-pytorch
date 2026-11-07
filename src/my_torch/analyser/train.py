"""--train mode of my_torch_analyzer."""

import os

import numpy as np

from my_torch.analyser.arguments import Arguments
from my_torch.analyser.network import load_network
from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import encode_all
from my_torch.encoding.labels import for_outputs
from my_torch.nn_module.neural_network_module import (
    EpochMetrics,
    Neural_network_module,
    TrainingHistory,
)


def train(args: Arguments) -> int:
    """Train LOADFILE on the labelled boards of CHESSFILE, then save it in
    SAVEFILE if given, otherwise back in LOADFILE. A 3-output network learns
    the states without colour, a 5-output one learns them with it; either
    width trains on either kind of chess file, since the side to move tells
    the colour of a label that does not carry one."""
    network = load_network(args.loadfile)
    with open(args.chessfile) as file:
        samples = read_samples(file, require_labels=True)
    if not samples:
        raise ValueError(f"{args.chessfile}: no board to train on")
    x = encode_all(sample.position for sample in samples)
    labels = for_outputs(network.layers[-1].output)
    y = labels.one_hot(
        [sample.label for sample in samples],
        [sample.position.side for sample in samples],
    )
    history = network.train(x, y, on_epoch=lambda metrics: _print_epoch(metrics, network.epochs))
    summary = _summary(network, history, x, y)
    _save(network, args.savefile or args.loadfile)
    print(f"Number of boards : {len(samples)}, {summary}")
    return 0


def _summary(network: Neural_network_module, history: TrainingHistory, x, y) -> str:
    """How good the saved network is. With validation, its accuracy on boards
    it never trained on; without, only its accuracy on the training boards,
    which cannot tell learning from learning by heart."""
    if history.best is not None:
        return (f"kept epoch {history.best.epoch} of {len(history)}, "
                f"{history.best.validation_accuracy:.1%} validation accuracy")
    accuracy = np.mean(network.forward_in_batches(x).argmax(axis=1) == y.argmax(axis=1))
    return f"{accuracy:.1%} accuracy on the training boards (no validation)"


def _print_epoch(metrics: EpochMetrics, epochs: int) -> None:
    """One line per epoch, flushed at once so a pipe sees it live."""
    line = f"epoch {metrics.epoch}/{epochs}: loss {metrics.loss:.4f}"
    if metrics.validation_loss is not None:
        line += (f", validation loss {metrics.validation_loss:.4f}"
                 f", validation accuracy {metrics.validation_accuracy:.1%}")
    print(line, flush=True)


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
