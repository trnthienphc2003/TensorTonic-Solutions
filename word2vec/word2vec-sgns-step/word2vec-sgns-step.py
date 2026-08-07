import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """

    # clone to avoid modifying inputs
    W_in = W_in.clone()
    W_out = W_out.clone()

    v_c = W_in[center_id]
    u_pos = W_out[pos_id]
    u_neg = W_out[neg_ids]   # (K, D)

    # ---- compute coefficients ----

    pos_score = torch.dot(u_pos, v_c)
    pos_coef = torch.sigmoid(pos_score) - 1

    neg_scores = torch.mv(u_neg, v_c)
    neg_coef = torch.sigmoid(neg_scores)   # (K,)

    # ---- gradients ----

    # center gradient
    grad_vc = pos_coef * u_pos + torch.sum(
        neg_coef[:, None] * u_neg,
        dim=0
    )

    # output gradients
    grad_u_pos = pos_coef * v_c

    grad_u_neg = neg_coef[:, None] * v_c


    # ---- SGD update ----

    W_in[center_id] -= lr * grad_vc

    W_out[pos_id] -= lr * grad_u_pos

    W_out.index_add_(0, neg_ids, -lr * grad_u_neg)


    return W_in, W_out