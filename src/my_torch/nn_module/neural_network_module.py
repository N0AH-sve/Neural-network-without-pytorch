import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from my_torch.nn_module.layer import Layer

EPOCH = 20
LEARNING_RATE = 0.005
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
PATIENCE = 3
SEED = 42 # everyone have 42 <
FORMAT_VERSION = 1

@dataclass(frozen=True)
class EpochMetrics:
    """What one training epoch measured; validation fields are None without
    validation samples."""
    epoch: int
    loss: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None


class TrainingHistory(list[EpochMetrics]):
    """The metrics of every epoch run, in order, and best: the epoch whose
    weights were kept (None without validation)."""
    best: EpochMetrics | None = None


class _EarlyStopping:
    """Keeps the weights of the epoch with the lowest validation loss, and
    tells when patience epochs in a row have not improved on it."""

    def __init__(self, patience : int):
        self.patience = patience
        self.best = None
        self.best_weights = None
        self.waited = 0

    def record(self, metrics : EpochMetrics, layers : list[Layer]) -> bool:
        """Take in an epoch, keeping its weights if it is the best so far;
        True once training should stop. Epochs without validation are ignored."""
        if metrics.validation_loss is None:
            return False
        if self.best is None or metrics.validation_loss < self.best.validation_loss:
            self.best, self.waited = metrics, 0
            self.best_weights = [(layer.w.copy(), layer.b.copy()) for layer in layers]
            return False
        self.waited += 1
        return self.waited >= self.patience

    def restore(self, layers : list[Layer]) -> None:
        """Put the best weights back; does nothing if no epoch was validated."""
        if self.best_weights is None:
            return
        for layer, (w, b) in zip(layers, self.best_weights, strict=True):
            layer.w, layer.b = w, b


# choice Active_fct between sigmoid fcttanh, relU, soft plus expo linera https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/
# choice weight there is a lot of method


class Neural_network_module:
    def __init__(self, int_layers : list[int] | None = None, epochs : int = EPOCH,
                 learning_rate : float = LEARNING_RATE, seed : int = SEED,
                 layers : list[Layer] | None = None, batch_size : int = BATCH_SIZE,
                 validation_split : float = VALIDATION_SPLIT, patience : int = PATIENCE):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.layers = self._init_layers(int_layers) if layers is None else layers
        self.version = 1


    def _init_layers(self, int_layers : list[int]) -> list[Layer]:
        layers = []
        for i in range(len(int_layers) - 1):
            next_inputs = int_layers[i + 1]
            activation = "linear" if i == len(int_layers) - 2 else "relu"
            layer = Layer(int_layers[i], next_inputs, activation)
            layers.append(layer)
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_in_batches(self, x, batch_size : int = 4096):
        """Same result as forward(x), computed on slices of batch_size samples:
        only one slice at a time is turned into floats, so a large boolean
        input never needs its full float64 copy in memory."""
        if len(x) == 0:
            return self.forward(x)
        return np.concatenate([self.forward(x[start:start + batch_size])
                               for start in range(0, len(x), batch_size)])

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, pas):
        for layer in self.layers:
            layer.update(pas)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "hyperparameters": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "validation_split": self.validation_split,
                "patience": self.patience,
            },
            "data": {
                "layers": [
                    {
                        "input": layer.input,
                        "output": layer.output,
                        "activation": layer.name_active_fct,
                        "w": layer.w.tolist(),
                        "b": layer.b.tolist(),
                    }
                    for layer in self.layers
                ]
            },
        }

    def save(self, path : str) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file)

    @staticmethod
    def _hyperparameters(raw : dict) -> dict:
        """Hyperparameters of a file or config; absent ones fall back to the constants."""
        hyper = raw.get("hyperparameters", {})
        return {
            "epochs": hyper.get("epochs", EPOCH),
            "learning_rate": hyper.get("learning_rate", LEARNING_RATE),
            "seed": hyper.get("seed", SEED),
            "batch_size": hyper.get("batch_size", BATCH_SIZE),
            "validation_split": hyper.get("validation_split", VALIDATION_SPLIT),
            "patience": hyper.get("patience", PATIENCE),
        }

    @classmethod
    def from_config(cls, conf : dict) -> "Neural_network_module":
        """Build a freshly randomized network from a validated generator config."""
        sizes, names = conf["layers"], conf["activations"]
        layers = [Layer(sizes[i], sizes[i + 1], names[i])
                  for i in range(len(sizes) - 1)]
        return cls(layers=layers, **cls._hyperparameters(conf))

    @staticmethod
    def _layer_from_dict(index : int, raw : dict) -> Layer:
        for key in ("input", "output", "activation", "w", "b"):
            if key not in raw:
                raise ValueError(f"layer {index}: missing key '{key}'")
        layer = Layer(raw["input"], raw["output"], raw["activation"],
                      raw["w"], raw["b"])
        if layer.w.shape != (layer.input, layer.output):
            raise ValueError(f"layer {index}: w has shape {layer.w.shape}, "
                             f"expected {(layer.input, layer.output)}")
        if layer.b.shape != (layer.output,):
            raise ValueError(f"layer {index}: b has shape {layer.b.shape}, "
                             f"expected {(layer.output,)}")
        return layer

    @classmethod
    def load(cls, path : str) -> "Neural_network_module":
        with open(path) as file:
            raw = json.load(file)

        if raw.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {raw.get('version')}")

        raw_layers = raw.get("data", {}).get("layers")
        if not raw_layers:
            raise ValueError("no layer found in data")

        layers = [cls._layer_from_dict(i, r) for i, r in enumerate(raw_layers)]
        for i, (current, following) in enumerate(zip(layers, layers[1:])):
            if current.output != following.input:
                raise ValueError(f"layer {i} outputs {current.output} but "
                                 f"layer {i + 1} expects {following.input}")

        return cls(layers=layers, **cls._hyperparameters(raw))

    def train(self, x, y,
              on_epoch : Callable[[EpochMetrics], None] | None = None) -> TrainingHistory:
        """Mini-batch gradient descent on x, keeping validation_split of the
        samples aside to measure the network on data it never learns from.

        The validation samples are drawn once, then every epoch visits the
        other samples in a new order. Both orders come from the seed, so the
        same network trained on the same data always ends up with the same
        weights.

        Training stops early once the validation loss has not improved for
        patience epochs, and the weights of the best epoch are put back: the
        epochs after it were only learning the training samples by heart.
        Returns what each epoch measured, up to the one where it stopped;
        on_epoch, if given, also receives each epoch's metrics as soon as the
        epoch ends, e.g. to show progress.
        """
        rng = np.random.default_rng(self.seed)
        training, validation = self._split(len(x), rng)
        x_validation, y_validation = x[validation], y[validation]
        stopping = _EarlyStopping(self.patience)
        history = TrainingHistory()
        for epoch in range(1, self.epochs + 1):
            loss = self._train_epoch(x, y, rng.permutation(training))
            history.append(EpochMetrics(epoch, loss, *self._validate(x_validation, y_validation)))
            if on_epoch is not None:
                on_epoch(history[-1])
            if stopping.record(history[-1], self.layers):
                break
        stopping.restore(self.layers)
        history.best = stopping.best
        return history

    def _split(self, n : int, rng : np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Training and validation indices: validation_split of the n samples
        are kept aside, but at least one is always left to train on."""
        order = rng.permutation(n)
        size = max(0, min(round(n * self.validation_split), n - 1))
        return order[size:], order[:size]

    def _train_epoch(self, x, y, order : np.ndarray) -> float:
        """One update per batch of batch_size samples, taken in the given
        order; returns the mean squared error over the epoch."""
        squared_error = 0.0
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            squared_error += self._step(x[batch], y[batch])
        return squared_error / max(1, len(order) * self.layers[-1].output)

    def _step(self, x, y) -> float:
        """Forward, backward and update on one batch; returns its summed squared error."""
        error = self.forward(x) - y
        self.backward(error / len(x))
        self.update(self.learning_rate)
        return float(np.sum(error ** 2))

    def _validate(self, x, y) -> tuple[float | None, float | None]:
        """Mean squared error and accuracy on samples never trained on. The
        accuracy is the share of samples whose largest output is where the
        target is largest, which is meaningful for one-hot targets."""
        if len(x) == 0:
            return None, None
        output = self.forward_in_batches(x)
        accuracy = float(np.mean(output.argmax(axis=1) == y.argmax(axis=1)))
        return float(np.mean((output - y) ** 2)), accuracy
