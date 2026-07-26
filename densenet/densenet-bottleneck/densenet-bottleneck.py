import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5, conv_pad=0):
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
    H_x = F.conv2d(logits, conv_weight, padding=conv_pad, stride=1)
    return H_x
    # pass


def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    # YOUR CODE HERE
    y1 = composite_layer(
        x,
        bn1_gamma,
        bn1_beta,
        bn1_mean,
        bn1_var,
        conv1_weight,
        eps=eps,
        conv_pad=0
    )

    y2 = composite_layer(
        y1,
        bn2_gamma,
        bn2_beta,
        bn2_mean,
        bn2_var,
        conv2_weight,
        eps=eps,
        conv_pad=1
    )

    return y2
