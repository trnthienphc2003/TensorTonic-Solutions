import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    x = x / torch.sqrt((x.to(torch.float32) ** 2).mean(dim=-1, keepdim=True) + epsilon).to(x.dtype)
    return x * g
