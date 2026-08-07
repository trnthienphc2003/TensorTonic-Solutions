import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # YOUR CODE HERE
    return F.softplus(-torch.dot(center_vec, pos_vec)) + torch.sum(
        F.softplus(torch.einsum('j,kj->k', center_vec, neg_vecs)),
        dim=-1
    )
