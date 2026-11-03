"""Time one training epoch for several batch sizes and OpenBLAS thread counts.

    uv run python benchmarks/threads.py

OpenBLAS reads its thread count when numpy is imported, so each count runs in
its own process. The boards are random 0/1 vectors as sparse as real ones: the
time of a matrix product does not depend on the values. Writes the chart to
docs/images/blas_threads.png and prints the table for docs/benchmarks.md.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

N_BOARDS = 20_000
TOPOLOGY = [769, 128, 64, 3]
BATCH_SIZES = [64, 512, N_BOARDS]
REPEATS = 7
CHART = Path(__file__).resolve().parent.parent / "docs" / "images" / "blas_threads.png"


def thread_counts() -> list[int]:
    cores = os.cpu_count() or 1
    return sorted({n for n in (1, 2, 4, 8, 16) if n < cores} | {cores})


def worker() -> None:
    """Print {batch_size: median seconds per epoch} as JSON."""
    import numpy as np

    from my_torch.nn_module.neural_network_module import Neural_network_module

    rng = np.random.default_rng(0)
    x = (rng.random((N_BOARDS, TOPOLOGY[0])) < 0.04).astype(float)
    y = np.eye(TOPOLOGY[-1])[rng.integers(0, TOPOLOGY[-1], N_BOARDS)]
    timings = {}
    for batch_size in BATCH_SIZES:
        runs = []
        for _ in range(REPEATS):
            net = Neural_network_module(TOPOLOGY, epochs=1, learning_rate=0.01,
                                        batch_size=batch_size)
            start = time.perf_counter()
            net.train(x, y)
            runs.append(time.perf_counter() - start)
        timings[batch_size] = statistics.median(runs)
    print(json.dumps(timings))


def measure() -> dict[int, dict[int, float]]:
    results = {}
    for threads in thread_counts():
        env = os.environ | {"OPENBLAS_NUM_THREADS": str(threads)}
        out = subprocess.run([sys.executable, __file__, "--worker"], env=env,
                             capture_output=True, text=True, check=True).stdout
        results[threads] = {int(b): t for b, t in json.loads(out).items()}
        print(f"{threads:>2} threads: {results[threads]}", file=sys.stderr)
    return results


def table(results: dict[int, dict[int, float]]) -> str:
    def name(batch_size):
        return "full batch" if batch_size == N_BOARDS else f"batch {batch_size}"

    head = "| Threads | " + " | ".join(name(b) for b in BATCH_SIZES) + " |"
    rule = "| --- |" + " --- |" * len(BATCH_SIZES)
    rows = [f"| {threads} | " + " | ".join(f"{times[b]:.2f} s" for b in BATCH_SIZES) + " |"
            for threads, times in results.items()]
    return "\n".join([head, rule, *rows])


def chart(results: dict[int, dict[int, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    threads = list(results)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for batch_size, colour in zip(BATCH_SIZES, colours, strict=True):
        seconds = [results[n][batch_size] for n in threads]
        label = "full batch" if batch_size == N_BOARDS else f"batch {batch_size}"
        ax.plot(threads, seconds, color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor=surface, markeredgewidth=2, label=label)
        ax.annotate(label, (threads[-1], seconds[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xticks(threads, [str(n) for n in threads])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("OpenBLAS threads", color=muted)
    ax.set_ylabel("seconds per epoch (20k boards)", color=muted)
    ax.set_title("Small batches get slower with too many threads", loc="left",
                 color=ink, fontsize=12)
    ax.grid(axis="y", color=grid, linewidth=1)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(frameon=False, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, facecolor=surface)


def main() -> None:
    if "--worker" in sys.argv:
        worker()
        return
    results = measure()
    chart(results)
    print(table(results))
    print(f"\nchart: {CHART}", file=sys.stderr)


if __name__ == "__main__":
    main()
