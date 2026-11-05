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
| Mini-batches + validation + early stopping, MSE (part 2) | `[769, 128, 64, 3]` | 57 kept of 60 run | 79.2% |
| Same, softmax cross-entropy | `[769, 128, 64, 3]` | 24 kept of 27 run | **79.2%** |

Always answering `Nothing` would score 41%.

## Learning curves: cross-entropy against mean squared error

![Validation accuracy per epoch for both losses, from the same initial
weights. Cross-entropy climbs much faster and stops at epoch 27, keeping epoch
24 at 79.2% on the test set; mean squared error climbs slowly and stops at
epoch 68, keeping epoch 65 at 79.1%](images/loss_comparison.png)

**Protocol.** `uv run python benchmarks/learning_curves.py --compare` (after
`uv run python -m my_torch.tools.shuffle_dataset`). Both losses train **a copy
of the same initial network**: `[769, 128, 64, 3]` (ReLU, ReLU, linear), batch
size 64, learning rate 0.05, epoch limit 100, `validation_split` 0.2,
`patience` 3, seed 42. `dataset/mixed/train.txt` gives 572,444 boards: 457,955
to train on and 114,489 held out by `train()` to decide when to stop.
`dataset/mixed/validation.txt` (143,111 boards) is the test set, used once at
the end. 4 OpenBLAS threads, on the machine described in the threads section.

| Loss | Stopped at | Kept epoch | Validation acc. | Test acc. | Time |
| --- | --- | --- | --- | --- | --- |
| `cross_entropy` | 27 of 100 | 24 | 79.2% | **79.2%** | **162 s** |
| `mse` | 68 of 100 | 65 | 79.1% | 79.1% | 419 s |

The two losses are on different scales (cross-entropy starts near ln 3 ≈ 1.1,
the mean squared error near 0.3), so only the accuracies compare.

**Results.**

- **Same accuracy, 2.7 times sooner.** Cross-entropy reaches 79.2% in 24
  epochs, the mean squared error 79.1% in 65: the gap between them is noise,
  the speed is not. Cross-entropy is the loss built for picking one class out
  of several: its gradient `softmax(scores) - Y` stays large while an answer is
  wrong, where the squared error's gradient shrinks as scores get close.
- **Cross-entropy also overfits sooner.** Its validation loss stops falling
  around epoch 21 while its training loss keeps dropping (0.35 against 0.50 at
  the kept epoch), so early stopping ends it at 27.
- **Stopping watches the loss, and accuracy was still rising**: epoch 27 had
  79.6% validation accuracy, 0.4 points above the kept epoch 24. Cross-entropy
  also penalises over-confident answers, so its loss can flatten while the
  accuracy still creeps up; a larger `patience` could gain a few tenths.
- **Both plateau near 79%**: the limit now comes from the network more than the
  loss. Next levers: a bigger network, regularisation against the early
  overfitting, the learning rate.

![Cross-entropy run. Left: training loss falling to 0.33 while the validation
loss flattens around 0.50 from epoch 21; dashed line at the kept epoch 24.
Right: validation accuracy rising from 59% to about 79.5%](images/learning_curves_cross_entropy.png)

| Epoch | Training loss | Validation loss | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 0.9298 | 0.8525 | 59.0% |
| 5 | 0.6608 | 0.6624 | 69.3% |
| 10 | 0.5210 | 0.5693 | 74.5% |
| 15 | 0.4391 | 0.5168 | 77.5% |
| 20 | 0.3844 | 0.5106 | 78.5% |
| 21 | 0.3757 | 0.5029 | 78.8% |
| 22 | 0.3672 | 0.5058 | 78.8% |
| 23 | 0.3596 | 0.5059 | 79.0% |
| **24 (kept)** | **0.3523** | **0.5016** | **79.2%** |
| 25 | 0.3451 | 0.5069 | 79.1% |
| 26 | 0.3386 | 0.5077 | 79.3% |
| 27 | 0.3322 | 0.5039 | 79.6% |

![Mean squared error run. Left: training loss falling to 0.11 while the
validation loss flattens around 0.156; dashed line at the kept epoch 65.
Right: validation accuracy rising from 55% to about 79%](images/learning_curves_mse.png)

| Epoch | Training loss | Validation loss | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 0.2990 | 0.2784 | 54.9% |
| 10 | 0.2117 | 0.2174 | 67.5% |
| 20 | 0.1773 | 0.1913 | 72.4% |
| 30 | 0.1535 | 0.1772 | 75.3% |
| 40 | 0.1372 | 0.1687 | 76.8% |
| 50 | 0.1255 | 0.1631 | 77.9% |
| 60 | 0.1168 | 0.1590 | 78.8% |
| 63 | 0.1145 | 0.1583 | 78.8% |
| 64 | 0.1137 | 0.1583 | 78.8% |
| **65 (kept)** | **0.1130** | **0.1562** | **79.1%** |
| 66 | 0.1123 | 0.1586 | 78.8% |
| 67 | 0.1117 | 0.1566 | 79.2% |
| 68 | 0.1109 | 0.1573 | 79.0% |

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
