import torch

def per_head_muon(parameter, gradient, previous_momentum, num_heads, momentum_coefficient, learning_rate):
    """
    Returns: updated parameter, momentum, and per-head orthogonalized update.
    """
    (
        parameter, gradient, previous_momentum
    ) = map(
        lambda x: torch.as_tensor(x),
        (
            parameter, gradient, previous_momentum
        )
    )

    momentum = momentum_coefficient * previous_momentum + gradient
    N = momentum.shape[0]
    momentum = momentum.view(num_heads, N // num_heads, -1)
    heads_out = []
    for head_idx in range(num_heads):
        U, S, Vt = torch.linalg.svd(momentum[head_idx, :], full_matrices=False)
        O = torch.matmul(U, Vt)
        heads_out.append(O)

    O = torch.cat(heads_out, dim=0)
    new_parameter = parameter - learning_rate * O
    momentum = momentum.view(N, -1)
    return new_parameter, momentum, O