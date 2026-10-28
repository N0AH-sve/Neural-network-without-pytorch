"""--train mode of my_torch_analyzer."""

from my_torch.analyser.arguments import Arguments
from my_torch.analyser.network import load_network
from my_torch.encoding.chessfile import read_samples
from my_torch.encoding.features import encode_all
from my_torch.encoding.labels import BASIC


def train(args: Arguments) -> int:
    """Train LOADFILE on the labelled boards of CHESSFILE, then save it in
    SAVEFILE if given, otherwise back in LOADFILE."""
    network = load_network(args.loadfile)
    with open(args.chessfile) as file:
        samples = read_samples(file, require_labels=True)
    if not samples:
        raise ValueError(f"{args.chessfile}: no board to train on")
    x = encode_all(sample.position for sample in samples)
    y = BASIC.one_hot(sample.label for sample in samples)
    network.train(x, y)
    network.save(args.savefile or args.loadfile)
    return 0
