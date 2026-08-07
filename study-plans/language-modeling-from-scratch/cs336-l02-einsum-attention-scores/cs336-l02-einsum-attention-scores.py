import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    B, S_q, D = q.shape
    S_k = k.shape[1]
    d_h = D // num_heads
    q = q.view(B, S_q, num_heads, d_h).transpose(1, 2)
    k = k.view(B, S_k, num_heads, d_h).transpose(1, 2)
    
    qk = torch.einsum('bhir,bhjr->bhij', q, k)
    return qk / (d_h ** .5)
    