# Benchmarks

> A benchmark without any visual proof is not worth anything.

## Protocol

To be filled: dataset size, train/validation/test split, metrics (accuracy,
precision, recall, confusion matrix), hardware.

## Results

Accuracy on the 143,111 boards of `dataset/mixed/validation.txt`, which no
network below was trained on or stopped with.

| Training | Topology | Epochs | Test acc. |
| --- | --- | --- | --- |
| Full batch, before mini-batches | `[769, 128, 64, 3]` | 500 on 20k boards | ~42% |
| Mini-batches (64), no validation | `[769, 128, 64, 3]` | 20 | 74.2% |
| Mini-batches + validation + early stopping | `[769, 128, 64, 3]` | 57 kept of 60 run | **79.2%** |

Always answering `Nothing` would score 41%.

## Learning curves

![Two panels over 60 epochs. Left, loss: the training loss keeps falling to
0.076 while the validation loss flattens around 0.104; a dashed line marks the
kept epoch 57. Right, validation accuracy rising from 55% to about 79%, the
same dashed line at epoch 57](images/learning_curves.png)

**Protocol.** `uv run python benchmarks/learning_curves.py` (after
`uv run python -m my_torch.tools.shuffle_dataset`). Network `[769, 128, 64, 3]`
(ReLU, ReLU, linear), mean squared error, batch size 64, learning rate 0.05,
epoch limit 100, `validation_split` 0.2, `patience` 3, seed 42.
`dataset/mixed/train.txt` gives 572,444 boards: 457,955 to train on and
114,489 held out by `train()` to decide when to stop. `dataset/mixed/validation.txt`
(143,111 boards) is the test set, used once at the end. 350 s on the machine
described in the threads section, 4 OpenBLAS threads.

| Epoch | Training loss | Validation loss | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 0.1992 | 0.1857 | 54.7% |
| 10 | 0.1393 | 0.1435 | 68.0% |
| 20 | 0.1172 | 0.1277 | 72.5% |
| 30 | 0.1012 | 0.1168 | 75.6% |
| 40 | 0.0900 | 0.1111 | 77.2% |
| 50 | 0.0818 | 0.1065 | 78.5% |
| 56 | 0.0779 | 0.1038 | 79.3% |
| **57 (kept)** | **0.0773** | **0.1034** | **79.3%** |
| 58 | 0.0767 | 0.1054 | 78.9% |
| 59 | 0.0762 | 0.1038 | 79.1% |
| 60 | 0.0756 | 0.1038 | 79.5% |

**Results.**

- **79.2% on the test set**, against 79.3% on the held-out validation boards:
  the validation estimate is trustworthy, nothing leaked between the sets.
- **The two losses part ways.** Until epoch ~5 they fall together; after that
  the training loss keeps falling (0.077 at epoch 57) while the validation loss
  flattens around 0.104. The widening gap is the network starting to learn the
  training boards by heart rather than rules that carry over.
- **Early stopping caught it**: epochs 58 to 60 did not beat epoch 57's
  validation loss, so training stopped at 60 of the 100 allowed and kept
  epoch 57's weights.
- **Stopping watches the loss, not the accuracy**: epoch 60 reached 79.5%
  validation accuracy, 0.2 points above the kept epoch, but with a higher
  loss. Accuracy moves by a few tenths from epoch to epoch; the loss is the
  smoother signal.
- **The plateau is shallow**: the validation loss was still creeping down by
  ~0.001 per few epochs before it stopped. A larger `patience` would train a
  little longer for a small gain; part 3 (softmax + cross-entropy) and the
  network size are more likely to move the curve itself.

## Training speed: batch size and OpenBLAS threads

numpy hands every matrix product to OpenBLAS, which splits it over several
threads. This measures how long one training epoch takes depending on the
batch size and on the number of threads.

![Seconds per epoch against OpenBLAS threads, one line per batch size: every
batch size is fastest at 4 threads; batch 64 is 2-2.5 times slower at 22
threads](images/blas_threads.png)

| Threads | batch 64 | batch 512 | full batch |
| --- | --- | --- | --- |
| 1 | 0.40 s | 0.31 s | 0.37 s |
| 2 | 0.28 s | 0.22 s | 0.27 s |
| 4 | **0.25 s** | **0.17 s** | **0.20 s** |
| 8 | 0.33 s | 0.20 s | 0.23 s |
| 16 | 0.43 s | 0.22 s | 0.26 s |
| 22 | 0.60 s | 0.43 s | 0.32 s |

**Protocol.** `uv run python benchmarks/threads.py`. One epoch of
`Neural_network_module.train` on 20,000 boards, network `[769, 128, 64, 3]`,
median of 7 runs per point; each thread count runs in its own process, since
OpenBLAS reads `OPENBLAS_NUM_THREADS` when numpy is imported. The boards are
random 0/1 vectors as sparse as real ones (4% ones): the time of a matrix
product does not depend on the values. Intel Core Ultra 7 155H (22 logical
cores), Python 3.14, numpy 2.5, OpenBLAS 0.3.34. The machine was not idle (load
average ~4): points vary by up to ~0.1 s between runs, the 16-thread one most.

**Results.**

- **4 threads is the fastest for every batch size.** Past that, threads spend
  longer coordinating than computing.
- **Small batches suffer most from too many threads**: batch 64 takes 2-2.5×
  longer at 22 threads than at 4, because its matrices (64 × 769 times
  769 × 128) are too small to split usefully. The full batch, with
  20,000-row matrices, barely changes.
- **Batch 512 is the fastest per epoch**, but makes 8 times fewer weight
  updates than batch 64: speed per epoch is not learning per epoch, which the
  accuracy benchmarks have to settle.

**Consequence.** The `./my_torch_analyzer` wrapper written by the Makefile sets
`OPENBLAS_NUM_THREADS=4` unless it is already set
(`BLAS_THREADS` in the Makefile). On the real `--train` with 20,000 boards and
batch 64, one run went from 22.1 s at 22 threads to 7.5 s at 4.

**Why not one thread per batch?** Mini-batch descent is sequential: each batch
starts from the weights the previous one produced. Splitting one batch across
threads and summing their gradients is what OpenBLAS already does inside each
matrix product.
