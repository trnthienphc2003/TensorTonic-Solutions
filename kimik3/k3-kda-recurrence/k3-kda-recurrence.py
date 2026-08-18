import torch
import torch.nn.functional as F

def kda_recurrence(query, key, value, decay_logits, write_strength, output_gate_logits, output_projection, initial_state, g_min=-5.0, eps=1e-6):
    """
    Returns: sequence outputs and the final recurrent state.
    """
    (
        query,
        key,
        value,
        decay_logits,
        write_strength,
        output_gate_logits,
        output_projection,
        initial_state,
    ) = map(
        torch.as_tensor,
        [
            query,
            key,
            value,
            decay_logits,
            write_strength,
            output_gate_logits,
            output_projection,
            initial_state,
        ],
    )

    B, N, H, d_k = query.shape
    _, _, _, d_v = value.shape

    # (B, N, H, d_k)
    alpha = torch.exp(g_min * torch.sigmoid(decay_logits))
    S = initial_state.clone()
    I = torch.eye(
        d_k,
        dtype=query.dtype,
        device=query.device
    ).view(1, 1, d_k, d_k)

    outs = []
    for t in range(N):
        beta_t = write_strength[:, t]
        k_t = key[:, t]                  # (B,H,d_k)
        q_t = query[:, t]                # (B,H,d_k)
        v_t = value[:, t]                # (B,H,d_v)
        alpha_t = alpha[:, t]            # (B,H,d_k)
    
        while beta_t.ndim < 4:
            beta_t = beta_t.unsqueeze(-1)
    
        # (B,H,d_k,d_k)
        kkT = (
            k_t.unsqueeze(-1)
            @ k_t.unsqueeze(-2)
        )
    
        # diag(alpha_t) @ S
        # (B,H,d_k,d_v)
        decayed_S = alpha_t.unsqueeze(-1) * S
    
        # (B,H,d_k,d_v)
        retained = (
            I - beta_t * kkT
        ) @ decayed_S
    
        # (B,H,d_k,d_v)
        kvT = (
            k_t.unsqueeze(-1)
            @ v_t.unsqueeze(-2)
        )
    
        # S_t
        S = retained + beta_t * kvT
    
        # S_t^T q_t
        # (B,H,d_v)
        h_t = (
            S.transpose(-1, -2)
            @ q_t.unsqueeze(-1)
        ).squeeze(-1)
        # h_t = F.normalize(h_t, dim=1)

        rms = torch.sqrt(
            torch.mean(h_t * h_t, dim=-1, keepdim=True)
            + eps
        )
    
        h_t = h_t / rms
    
        # output gate
        gate_t = torch.sigmoid(
            output_gate_logits[:, t]
        )
    
        h_t = gate_t * h_t
    
        # concatenate heads
        h_t = h_t.reshape(B, H * d_v)
    
        # project to model width
        o_t = h_t @ output_projection.transpose(0, 1)
        # o_t = F.normalize(o_t, dim=1)
        outs.append(o_t)
        # S_t = (I - beta_t)
    
    # assert False, S_t.shape
    outs = torch.stack(outs, dim=1)
    return outs, S