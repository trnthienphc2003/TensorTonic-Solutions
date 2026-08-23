import math
import torch

def gated_mla(hidden_states, query_projection, latent_down_projection, key_up_projection, value_up_projection, output_gate_projection, output_projection, num_heads, causal=True):
    """
    Returns: gated attention outputs and the latent key-value cache.
    """
    pass
    (
        hidden_states, 
        query_projection, 
        latent_down_projection, 
        key_up_projection, 
        value_up_projection, 
        output_gate_projection, 
        output_projection
    ) = map(
        lambda x: torch.as_tensor(x),
        [
            hidden_states, 
            query_projection, 
            latent_down_projection, 
            key_up_projection, 
            value_up_projection, 
            output_gate_projection, 
            output_projection
        ]
    )

    B, S, D = hidden_states.shape
    assert D % num_heads == 0
    D_h = D // num_heads
    # hidden_states = hidden_states.transpose(1, 2)

    # assert False, query_projection.shape
    Q = hidden_states @ query_projection.transpose(0, 1)
    C = hidden_states @ latent_down_projection.transpose(0, 1)
    K = C @ key_up_projection.transpose(0, 1)
    V = C @ value_up_projection.transpose(0, 1)
    Q = Q.contiguous().view(B, S, num_heads, D_h).transpose(1, 2)
    K = K.contiguous().view(B, S, num_heads, D_h).transpose(1, 2)
    V = V.contiguous().view(B, S, num_heads, D_h).transpose(1, 2)

    A = Q @ K.transpose(-1, -2) / (D_h ** .5)
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=A.device), 1)
        A = A.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(A, dim=-1)
    context = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S, D)
    # assert False
    gate = torch.sigmoid(hidden_states @ output_gate_projection.transpose(0, 1))
    out = (context * gate) @ output_projection.transpose(0, 1)

    return out, C