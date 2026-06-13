import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    x = np.asarray(x_T, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    epsilon_preds = np.asarray(epsilon_preds, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)

    T = len(betas)

    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    for step, t in enumerate(range(T - 1, -1, -1)):
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        eps = epsilon_preds[step]

        x = (1.0 / np.sqrt(alpha_t)) * (
            x - beta_t / np.sqrt(1.0 - alpha_bar_t) * eps
        )

        if t > 0:
            x += np.sqrt(beta_t) * z_values[step]

    return x