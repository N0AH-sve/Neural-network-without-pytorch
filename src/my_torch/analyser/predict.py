import numpy as np

from my_torch.encoding.chessfile import Sample, read_samples
from my_torch.encoding.features import INPUT_SIZE, encode_all
from my_torch.encoding.labels import BASIC, FULL, LabelSet
from my_torch.nn_module.neural_network_module import Neural_network_module

LABEL_SETS = {len(BASIC): BASIC, len(FULL): FULL}


class Predict:
    def __init__(self, loadfile: str, chessfile: str):
        self.loadfile = loadfile
        self.chessfile = chessfile

    @staticmethod
    def label_set_of(nn: Neural_network_module) -> LabelSet:
        """Pick the label set from the width of the output layer.

        The file format stores no class names, so the number of output neurons
        is the only hint: 3 means plain states, 5 means states with a colour.
        """
        width = nn.layers[-1].output
        try:
            return LABEL_SETS[width]
        except KeyError:
            valid = ", ".join(str(size) for size in sorted(LABEL_SETS))
            raise ValueError(
                f"{width} output neurons, expected one of: {valid}"
            ) from None

    @staticmethod
    def classify(net: Neural_network_module, samples: list[Sample]) -> np.ndarray:
        """Index of the winning output neuron for each sample, in order.

        Every board goes through a single forward pass: argmax is unchanged by
        a softmax, so the raw scores are enough to rank the classes.
        """
        x = encode_all(sample.position for sample in samples)
        scores = net.forward(x)
        return scores.argmax(axis=1)

    def run(self) -> int:
        """Print the state of every board of the chess file, one per line.

        The whole file is parsed before the first line is printed: a broken
        board must not leave a truncated prediction list behind.
        """
        net = Neural_network_module.load(self.loadfile)
        labels = self.label_set_of(net)
        if net.layers[0].input != INPUT_SIZE:
            raise ValueError(
                f"network takes {net.layers[0].input} inputs, "
                f"expected {INPUT_SIZE}"
            )
        with open(self.chessfile) as file:
            samples = read_samples(file)
        for index in self.classify(net, samples):
            print(labels.label(index))
        return 0
