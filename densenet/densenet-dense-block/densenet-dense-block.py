import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    # YOUR CODE HERE
    x, bn_mean, bn_var, bn_gamma, bn_beta, conv_weight = map(lambda t: torch.tensor(t, dtype=torch.float64), (x, bn_mean, bn_var, bn_gamma, bn_beta, conv_weight))
    norm_x = F.batch_norm(
        input=x, 
        running_mean=bn_mean, 
        running_var=bn_var, 
        weight=bn_gamma, 
        bias=bn_beta, 
        eps=eps
    )
    logits = F.relu(norm_x)
    H_x = F.conv2d(logits, conv_weight, padding=1, stride=1)
    return H_x
    # pass

def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    # YOUR CODE HERE
    x = torch.Tensor(x).to(float)
    for layer in layers:
        new_feats = composite_layer(
            x,
            layer["bn_gamma"],
            layer["bn_beta"],
            layer["bn_mean"],
            layer["bn_var"],
            layer["conv_weight"],
            eps=eps
        )

        x = torch.cat([x, new_feats], dim=1)

    return x