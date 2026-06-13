import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    return np.linspace(beta_1, beta_T, T)

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    t = np.arange(1, T + 1, dtype=np.float64)

    alpha_bars = np.cos(
        ((t / T + s) / (1 + s)) * np.pi / 2
    ) ** 2

    alpha_bars /= np.cos(
        (s / (1 + s)) * np.pi / 2
    ) ** 2

    alpha_bars = np.clip(alpha_bars, 0.0001, 0.9999)
    return np.round(alpha_bars, 6).tolist()

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    alpha_bars = np.asarray(alpha_bars, dtype=np.float64)

    betas = np.empty_like(alpha_bars)
    betas[0] = 1.0 - alpha_bars[0]
    betas[1:] = 1.0 - alpha_bars[1:] / alpha_bars[:-1]

    return np.clip(betas, 0.0001, 0.9999)