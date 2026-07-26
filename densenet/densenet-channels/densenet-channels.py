import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE
    pass
    lst = [stem_channels]
    for i in range(len(block_layers)):
        lst.append(lst[-1] + int(block_layers[i] * growth_rate))
        if i != len(block_layers) - 1:
            lst.append(math.floor(lst[-1] * compression))

    return torch.Tensor(lst).to(int)