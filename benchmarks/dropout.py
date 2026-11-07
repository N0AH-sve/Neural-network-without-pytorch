"""Train the same network at several dropout rates and plot what it changes.

    uv run python -m my_torch.tools.shuffle_dataset   # once: dataset/mixed/
    OPENBLAS_NUM_THREADS=4 uv run python benchmarks/dropout.py \
        [--epochs N] [--rates 0 0.2 0.5] [--train FILE] [--test FILE]

Every rate starts from the same initial weights and sees the same boards in
the same order, so the only thing that differs is which units are dropped.

What dropout is supposed to buy is not a lower training loss -- it is a
smaller gap between the training loss and the validation loss. The loss that
train() reports per epoch is measured on a dropped sub-network, so it runs
higher than the whole network's and cannot be compared across rates: the gap
below is recomputed at the end with a dropout-free pass over the same boards.

Writes docs/images/dropout.png and prints the table for docs/benchmarks.md.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import INPUT_SIZE, encode_all
from my_torch.encoding.labels import for_outputs
from my_torch.nn_module import losses
from my_torch.nn_module.neural_network_module import (
    EpochMetrics,
    Neural_network_module,
    TrainingHistory,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
MIXED = ROOT / "dataset" / "mixed"
OUTPUTS = 5
TOPOLOGY = [INPUT_SIZE, 128, 64, OUTPUTS]

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = ["#2a78d6", "#eb6834", "#2e9e6b", "#9b59b6", "#c0392b"]


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Boards and one-hot targets of a chess file, in its own order."""
    if not path.exists():
        raise SystemExit(f"{path}: no such file. Build the dataset first:\n"
                         f"    uv run python -m my_torch.tools.shuffle_dataset")
    with open(path) as file:
        samples = read_samples(file, require_labels=True)
    labels = for_outputs(OUTPUTS)
    x = encode_all(s.position for s in samples)
    y = labels.one_hot([s.label for s in samples],
                       [s.position.side for s in samples])
    return x, y


def whole_network_loss(network: Neural_network_module, x, y) -> float:
    """Loss of the network with every unit in place, which is the only loss
    comparable between rates: train() reports a dropped sub-network's."""
    return float(losses.get(network.loss).value(network.forward_in_batches(x), y))


def accuracy(network: Neural_network_module, x, y) -> float:
    return float(np.mean(network.forward_in_batches(x).argmax(axis=1) == y.argmax(axis=1)))


class Result:
    """What one rate ended up with."""

    def __init__(self, rate: float, history: TrainingHistory,
                 network: Neural_network_module, x, y, x_test, y_test, seconds: float):
        self.rate = rate
        self.history = history
        self.seconds = seconds
        self.training_loss = whole_network_loss(network, x, y)
        self.validation_loss = history.best.validation_loss
        self.gap = self.validation_loss - self.training_loss
        self.training_accuracy = accuracy(network, x, y)
        self.test_accuracy = accuracy(network, x_test, y_test)

    @property
    def row(self) -> str:
        best = self.history.best
        return (f"| {self.rate} | {best.epoch} of {len(self.history)} "
                f"| {self.training_loss:.4f} | {self.validation_loss:.4f} "
                f"| {self.gap:+.4f} | {self.training_accuracy:.1%} "
                f"| {self.test_accuracy:.1%} | {self.seconds:.0f}s |")


def table(results: list[Result]) -> str:
    rows = [("| Dropout | Kept epoch | Training loss | Validation loss | Gap "
             "| Training accuracy | Test accuracy | Time |"),
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows += [r.row for r in results]
    return "\n".join(rows)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("epoch", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def chart(results: list[Result]) -> Path:
    """Validation accuracy and validation loss per epoch for each rate: the
    accuracy says whether dropout helped, the loss curve says whether it
    stopped the network from turning on itself."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150,
                                               facecolor=SURFACE)
    for result, colour in zip(results, COLOURS):
        epochs = [m.epoch for m in result.history]
        label = f"dropout {result.rate}: test {result.test_accuracy:.1%}"
        accuracy_ax.plot(epochs, [100 * m.validation_accuracy for m in result.history],
                         color=colour, linewidth=2, label=label)
        loss_ax.plot(epochs, [m.validation_loss for m in result.history],
                     color=colour, linewidth=2, label=f"dropout {result.rate}")
        best = result.history.best
        accuracy_ax.plot(best.epoch, 100 * best.validation_accuracy, "o", color=colour,
                         markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)

    accuracy_ax.set_title("Validation accuracy (%)", loc="left", color=INK, fontsize=11)
    accuracy_ax.legend(frameon=False, labelcolor=INK, loc="lower right", fontsize=9)
    loss_ax.set_title("Validation loss", loc="left", color=INK, fontsize=11)
    loss_ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    for ax in (accuracy_ax, loss_ax):
        _style(ax)
    fig.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / "dropout.png"
    fig.savefig(path, facecolor=SURFACE)
    return path


def run(rate: float, start: Neural_network_module, x, y, x_test, y_test) -> Result:
    network = copy.deepcopy(start)
    network.set_dropout(rate)

    def progress(m: EpochMetrics) -> None:
        print(f"[dropout {rate}] epoch {m.epoch}: loss {m.loss:.4f}, validation loss "
              f"{m.validation_loss:.4f}, validation accuracy {m.validation_accuracy:.1%}",
              flush=True)

    began = time.perf_counter()
    history = network.train(x, y, on_epoch=progress)
    result = Result(rate, history, network, x, y, x_test, y_test,
                    time.perf_counter() - began)
    print(f"\n[dropout {rate}] kept epoch {history.best.epoch} of {len(history)}; "
          f"training loss {result.training_loss:.4f}, validation loss "
          f"{result.validation_loss:.4f} (gap {result.gap:+.4f}); "
          f"test accuracy {result.test_accuracy:.1%}; {result.seconds:.0f}s\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=100, help="epoch limit (default: 100)")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.2, 0.5],
                        help="dropout rates to compare (default: 0 0.2 0.5)")
    parser.add_argument("--train", type=Path, default=MIXED / "train.txt",
                        help="training boards (default: dataset/mixed/train.txt)")
    parser.add_argument("--test", type=Path, default=MIXED / "validation.txt",
                        help="test boards, used once (default: dataset/mixed/validation.txt)")
    args = parser.parse_args()

    x, y = load(args.train)
    x_test, y_test = load(args.test)
    print(f"{len(x)} training boards, {len(x_test)} test boards, "
          f"topology {TOPOLOGY}, up to {args.epochs} epochs\n")

    start = Neural_network_module(TOPOLOGY, epochs=args.epochs, learning_rate=0.05,
                                  batch_size=64, patience=3)
    results = [run(rate, start, x, y, x_test, y_test) for rate in args.rates]
    print(table(results))
    print(f"\nchart: {chart(results)}")


if __name__ == "__main__":
    main()
