import json

import numpy as np

from my_torch.nn_module.layer import Layer

EPOCH = 20
LEARNING_RATE = 0.005
BATCH_SIZE = 64
SEED = 42 # everyone have 42 <
FORMAT_VERSION = 1

# choice Active_fct between sigmoid fcttanh, relU, soft plus expo linera https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/
# choice weight there is a lot of method


class Neural_network_module:
    def __init__(self, int_layers : list[int] | None = None, epochs : int = EPOCH,
                 learning_rate : float = LEARNING_RATE, seed : int = SEED,
                 layers : list[Layer] | None = None, batch_size : int = BATCH_SIZE):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.batch_size = batch_size
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

    def train(self, x, y):
        """Mini-batch gradient descent: every epoch visits the samples once, in
        a new order, and updates the weights after each batch.

        The order comes from the seed, so the same network trained on the same
        data always ends up with the same weights. A batch_size of at least
        len(x) gives plain full-batch descent.
        """
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), self.batch_size):
                batch = order[start:start + self.batch_size]
                output = self.forward(x[batch])
                self.backward((output - y[batch]) / len(batch))
                self.update(self.learning_rate)
