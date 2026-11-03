"""Train on the mixed dataset and plot the training and validation curves.

    uv run python -m my_torch.tools.shuffle_dataset   # once: dataset/mixed/
    OPENBLAS_NUM_THREADS=4 uv run python benchmarks/learning_curves.py [--epochs N]

train() keeps validation_split of dataset/mixed/train.txt aside to decide when
to stop; dataset/mixed/validation.txt is used only once, at the end, as a test
set the network never saw. Writes the chart to docs/images/learning_curves.png
and prints the per-epoch table for docs/benchmarks.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import INPUT_SIZE, encode_all
from my_torch.encoding.labels import BASIC
from my_torch.nn_module.neural_network_module import (
    EpochMetrics,
    Neural_network_module,
    TrainingHistory,
)

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "docs" / "images" / "learning_curves.png"
TOPOLOGY = [INPUT_SIZE, 128, 64, len(BASIC)]


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as file:
        samples = read_samples(file, require_labels=True)
    return encode_all(s.position for s in samples), BASIC.one_hot(s.label for s in samples)


def chart(history: TrainingHistory) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    training, validation = "#2a78d6", "#eb6834"
    epochs = [m.epoch for m in history]
    kept = history.best

    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150,
                                               facecolor=surface)
    loss_ax.plot(epochs, [m.loss for m in history], color=training, linewidth=2,
                 label="training loss")
    loss_ax.plot(epochs, [m.validation_loss for m in history], color=validation,
                 linewidth=2, label="validation loss")
    loss_ax.set_title("Loss (mean squared error)", loc="left", color=ink, fontsize=11)
    loss_ax.legend(frameon=False, labelcolor=ink)
    accuracy_ax.plot(epochs, [100 * m.validation_accuracy for m in history],
                     color=validation, linewidth=2)
    accuracy_ax.set_title("Validation accuracy (%)", loc="left", color=ink, fontsize=11)

    for ax in (loss_ax, accuracy_ax):
        ax.set_facecolor(surface)
        ax.axvline(kept.epoch, color=muted, linewidth=1, linestyle="--")
        ax.annotate(f"kept: epoch {kept.epoch}", (kept.epoch, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                    color=muted, fontsize=9)
        ax.set_xlabel("epoch", color=muted)
        ax.grid(axis="y", color=grid, linewidth=1)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, facecolor=surface)


def table(history: TrainingHistory) -> str:
    rows = ["| Epoch | Training loss | Validation loss | Validation accuracy |",
            "| --- | --- | --- | --- |"]
    for m in history:
        mark = " (kept)" if m is history.best else ""
        rows.append(f"| {m.epoch}{mark} | {m.loss:.4f} | {m.validation_loss:.4f} "
                    f"| {m.validation_accuracy:.1%} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=100, help="epoch limit (default: 100)")
    args = parser.parse_args()

    start = time.perf_counter()
    x, y = load(ROOT / "dataset" / "mixed" / "train.txt")
    x_test, y_test = load(ROOT / "dataset" / "mixed" / "validation.txt")
    network = Neural_network_module(TOPOLOGY, epochs=args.epochs, learning_rate=0.05,
                                    batch_size=64, patience=3)

    def progress(m: EpochMetrics) -> None:
        print(f"epoch {m.epoch}: loss {m.loss:.4f}, validation loss {m.validation_loss:.4f}, "
              f"validation accuracy {m.validation_accuracy:.1%}", flush=True)

    history = network.train(x, y, on_epoch=progress)
    test = np.mean(network.forward_in_batches(x_test).argmax(axis=1) == y_test.argmax(axis=1))
    chart(history)
    print(f"\n{len(x)} boards ({round(len(x) * network.validation_split)} held out), "
          f"stopped after epoch {len(history)} of {args.epochs}, kept epoch "
          f"{history.best.epoch}; test accuracy {test:.1%} on {len(x_test)} boards; "
          f"{time.perf_counter() - start:.0f}s\n")
    print(table(history))
    print(f"\nchart: {CHART}")


if __name__ == "__main__":
    main()
