import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    N = counts.sum()
    freq = counts / N
    return torch.minimum((t / freq) ** .5, torch.ones_like(counts))
