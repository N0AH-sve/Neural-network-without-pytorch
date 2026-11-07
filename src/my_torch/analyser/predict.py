"""--predict mode of my_torch_analyzer."""

import numpy as np

from my_torch.analyser.network import load_network
from my_torch.encoding.chessfile import Sample, read_samples
from my_torch.encoding.features import encode_all
from my_torch.encoding.labels import for_outputs
from my_torch.nn_module.neural_network_module import Neural_network_module


class Predict:
    def __init__(self, loadfile: str, chessfile: str):
        self.loadfile = loadfile
        self.chessfile = chessfile

    @staticmethod
    def classify(net: Neural_network_module, samples: list[Sample]) -> np.ndarray:
        """Index of the winning output neuron for each sample, in order.

        Every board goes through a single forward pass: argmax is unchanged by
        a softmax, so the raw scores are enough to rank the classes. The pass
        is cut into batches, so a large chess file does not ask for every
        activation of every layer at once.
        """
        x = encode_all(sample.position for sample in samples)
        scores = net.forward_in_batches(x)
        return scores.argmax(axis=1)

    def run(self) -> int:
        """Print the state of every board of the chess file, one per line.

        The whole file is parsed before the first line is printed: a broken
        board must not leave a truncated prediction list behind.
        """
        net = load_network(self.loadfile)
        labels = for_outputs(net.layers[-1].output)
        with open(self.chessfile) as file:
            samples = read_samples(file)
        for index in self.classify(net, samples):
            print(labels.label(index))
        return 0
