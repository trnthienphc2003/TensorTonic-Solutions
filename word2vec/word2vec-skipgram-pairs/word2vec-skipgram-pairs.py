import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    N = len(token_ids)
    grid_x, grid_y = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    choose_x, choose_y = torch.where(
        (grid_x != grid_y) 
        & (torch.abs(grid_y - grid_x) <= window)
    )

    return torch.stack([token_ids[choose_x], token_ids[choose_y]], dim=-1)
    # assert False, f'{choose_x} {choose_y}'