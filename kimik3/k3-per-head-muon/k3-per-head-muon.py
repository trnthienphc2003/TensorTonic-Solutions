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
    momentum_heads = momentum.view(num_heads, N // num_heads, -1)
    U, S, Vt = torch.linalg.svd(momentum_heads, full_matrices=False)
    O_heads = torch.bmm(U, Vt)

    O = O_heads.view(N, -1)
    new_parameter = parameter - learning_rate * O
    return new_parameter, momentum, O