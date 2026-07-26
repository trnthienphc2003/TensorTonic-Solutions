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


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    # 1. Ensure input tensor is double precision (float64) matching weights
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    else:
        x = x.to(dtype=torch.float64)

    # 2. Initial Stem Convolution (usually 3x3 same-padding or 7x7 depending on dataset)
    stem_weight = torch.tensor(weights['stem_conv'], dtype=torch.float64) if not isinstance(weights['stem_conv'], torch.Tensor) else weights['stem_conv'].to(torch.float64)
    # Determine padding based on kernel size (assuming same padding)
    pad = stem_weight.shape[-1] // 2
    x = F.conv2d(x, stem_weight, padding=pad, stride=1)

    # 3. Alternate between Dense Blocks and Transition Layers
    num_blocks = len(weights['blocks'])
    for i in range(num_blocks):
        # Apply Dense Block
        x = dense_block(x, weights['blocks'][i], growth_rate, eps=eps)
        
        # Apply Transition Layer (if present for this block index)
        if i < len(weights['transitions']):
            trans = weights['transitions'][i]
            x = transition_layer(
                x,
                bn_gamma=trans['bn_gamma'],
                bn_beta=trans['bn_beta'],
                bn_mean=trans['bn_mean'],
                bn_var=trans['bn_var'],
                conv_weight=trans['conv_weight'],
                eps=eps
            )

    # 4. Final Batch Normalization + ReLU
    final_gamma = torch.tensor(weights['final_bn_gamma'], dtype=torch.float64)
    final_beta = torch.tensor(weights['final_bn_beta'], dtype=torch.float64)
    final_mean = torch.tensor(weights['final_bn_mean'], dtype=torch.float64)
    final_var = torch.tensor(weights['final_bn_var'], dtype=torch.float64)

    x = F.batch_norm(
        x,
        running_mean=final_mean,
        running_var=final_var,
        weight=final_gamma,
        bias=final_beta,
        eps=eps
    )
    x = F.relu(x)

    # 5. Global Average Pooling over Spatial Dimensions (H, W)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, start_dim=1)

    # 6. Fully Connected (Linear) Layer to produce Class Logits
    fc_weight = torch.tensor(weights['fc_weight'], dtype=torch.float64)
    fc_bias = torch.tensor(weights['fc_bias'], dtype=torch.float64)

    logits = F.linear(x, fc_weight, fc_bias)

    return logits
