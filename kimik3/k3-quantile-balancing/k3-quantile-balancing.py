import torch

def quantile_balancing(router_scores, current_bias, selected_count):
    """
    Returns: selected experts, mixture weights, loads, and the next centered bias.
    """
    T, E = router_scores.shape
    biased_scores = router_scores + current_bias
    ranked = torch.argsort(biased_scores, dim=-1, descending=True, stable=True)
    selected_exp = ranked[:, :selected_count]
    selected_exp_raw = router_scores.gather(1, selected_exp)

    mixture_w = selected_exp_raw / selected_exp_raw.sum(dim=-1, keepdim=True)
    loads = torch.bincount(selected_exp.reshape(-1), minlength=E)
    cutoff = biased_scores.gather(1, ranked[:, selected_count].unsqueeze(-1))
    target_load = (T * selected_count) // E

    margins = router_scores - cutoff
    ordered_margins = torch.sort(margins, dim=0, descending=True, stable=True).values
    uncentered_bias = -ordered_margins[target_load]
    next_bias = uncentered_bias - uncentered_bias.mean()
    return selected_exp, mixture_w, loads, next_bias