"""Train on the mixed dataset and plot the training and validation curves.

    uv run python -m my_torch.tools.shuffle_dataset   # once: dataset/mixed/
    OPENBLAS_NUM_THREADS=4 uv run python benchmarks/learning_curves.py \
        [--epochs N] [--loss mse|cross_entropy | --compare]

train() keeps validation_split of dataset/mixed/train.txt aside to decide when
to stop; dataset/mixed/validation.txt is used only once, at the end, as a test
set the network never saw. Writes docs/images/learning_curves_<loss>.png and
prints the per-epoch table for docs/benchmarks.md. --compare trains every loss
from the same initial weights and also writes docs/images/loss_comparison.png.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import INPUT_SIZE, encode_all
from my_torch.encoding.labels import BASIC
from my_torch.nn_module import losses
from my_torch.nn_module.neural_network_module import (
    EpochMetrics,
    Neural_network_module,
    TrainingHistory,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
TOPOLOGY = [INPUT_SIZE, 128, 64, len(BASIC)]


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as file:
        samples = read_samples(file, require_labels=True)
    return encode_all(s.position for s in samples), BASIC.one_hot(s.label for s in samples)


SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("epoch", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def chart(history: TrainingHistory, loss: str) -> Path:
    plt = _pyplot()
    surface, ink, muted = SURFACE, INK, MUTED
    training, validation = "#2a78d6", "#eb6834"
    epochs = [m.epoch for m in history]
    kept = history.best

    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150,
                                               facecolor=surface)
    loss_ax.plot(epochs, [m.loss for m in history], color=training, linewidth=2,
                 label="training loss")
    loss_ax.plot(epochs, [m.validation_loss for m in history], color=validation,
                 linewidth=2, label="validation loss")
    loss_ax.set_title(f"Loss ({loss})", loc="left", color=ink, fontsize=11)
    loss_ax.legend(frameon=False, labelcolor=ink)
    accuracy_ax.plot(epochs, [100 * m.validation_accuracy for m in history],
                     color=validation, linewidth=2)
    accuracy_ax.set_title("Validation accuracy (%)", loc="left", color=ink, fontsize=11)

    for ax in (loss_ax, accuracy_ax):
        _style(ax)
        ax.axvline(kept.epoch, color=muted, linewidth=1, linestyle="--")
        ax.annotate(f"kept: epoch {kept.epoch}", (kept.epoch, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                    color=muted, fontsize=9)
    fig.tight_layout()
    path = IMAGES / f"learning_curves_{loss}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    return path


def comparison_chart(results: dict[str, tuple[TrainingHistory, float]]) -> Path:
    """Validation accuracy per epoch for each loss: the loss values themselves
    are on different scales, so accuracy is what can be compared."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    for (loss, (history, test)), colour in zip(results.items(), ["#2a78d6", "#eb6834"]):
        epochs = [m.epoch for m in history]
        accuracy = [100 * m.validation_accuracy for m in history]
        ax.plot(epochs, accuracy, color=colour, linewidth=2,
                label=f"{loss}: kept epoch {history.best.epoch}, test {test:.1%}")
        ax.plot(history.best.epoch, 100 * history.best.validation_accuracy, "o",
                color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(loss, (epochs[-1], accuracy[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    _style(ax)
    ax.set_ylabel("validation accuracy (%)", color=MUTED)
    ax.set_title("Same network, same start: validation accuracy per loss", loc="left",
                 color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    path = IMAGES / "loss_comparison.png"
    fig.savefig(path, facecolor=SURFACE)
    return path


def table(history: TrainingHistory) -> str:
    rows = ["| Epoch | Training loss | Validation loss | Validation accuracy |",
            "| --- | --- | --- | --- |"]
    for m in history:
        mark = " (kept)" if m is history.best else ""
        rows.append(f"| {m.epoch}{mark} | {m.loss:.4f} | {m.validation_loss:.4f} "
                    f"| {m.validation_accuracy:.1%} |")
    return "\n".join(rows)


def run(network: Neural_network_module, x, y, x_test, y_test) -> tuple[TrainingHistory, float]:
    """Train network, print its progress, and return its history and test accuracy."""
    def progress(m: EpochMetrics) -> None:
        print(f"[{network.loss}] epoch {m.epoch}: loss {m.loss:.4f}, validation loss "
              f"{m.validation_loss:.4f}, validation accuracy {m.validation_accuracy:.1%}",
              flush=True)

    start = time.perf_counter()
    history = network.train(x, y, on_epoch=progress)
    test = float(np.mean(network.forward_in_batches(x_test).argmax(axis=1)
                         == y_test.argmax(axis=1)))
    print(f"\n[{network.loss}] {len(x)} boards ({round(len(x) * network.validation_split)} "
          f"held out), stopped after epoch {len(history)} of {network.epochs}, kept epoch "
          f"{history.best.epoch}; test accuracy {test:.1%} on {len(x_test)} boards; "
          f"{time.perf_counter() - start:.0f}s\n")
    print(table(history))
    print(f"\nchart: {chart(history, network.loss)}\n")
    return history, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=100, help="epoch limit (default: 100)")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--loss", choices=losses.names(), default="cross_entropy",
                        help="loss to train with (default: cross_entropy)")
    choice.add_argument("--compare", action="store_true",
                        help="train every loss from the same initial weights")
    args = parser.parse_args()

    x, y = load(ROOT / "dataset" / "mixed" / "train.txt")
    x_test, y_test = load(ROOT / "dataset" / "mixed" / "validation.txt")
    start = Neural_network_module(TOPOLOGY, epochs=args.epochs, learning_rate=0.05,
                                  batch_size=64, patience=3)
    results = {}
    for loss in losses.names() if args.compare else [args.loss]:
        network = copy.deepcopy(start)
        network.loss = loss
        results[loss] = run(network, x, y, x_test, y_test)
    if args.compare:
        print(f"comparison: {comparison_chart(results)}")


if __name__ == "__main__":
    main()
