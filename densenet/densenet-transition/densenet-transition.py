import torch
import torch.nn.functional as F


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
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
    z = F.conv2d(logits, conv_weight, padding=0, stride=1)
    out = F.avg_pool2d(z, kernel_size=2, ceil_mode=False)
    return out