def expected_calibration_error(y_true: list, y_pred: list, n_bins: int) -> float:
    """
    Returns the calibration error as a float.
    """
    # Write code here
    bin_r = [(i + 1) / n_bins for i in range(n_bins)]
    bin_cnt = [0 for _ in range(n_bins)]
    bin_size = bin_cnt[:]
    bin_conf = [0.0 for _ in range(n_bins)]
    for i in range(len(y_true)):
        idx_bin = min(int(y_pred[i] * n_bins), n_bins - 1)
        # assert False, idx_bin
        bin_cnt[idx_bin] += y_true[i]
        bin_conf[idx_bin] += y_pred[i]
        bin_size[idx_bin] += 1

    ECE = 0.0
    bin_acc = [bin_cnt[i] / bin_size[i] if bin_size[i] != 0 else 0.0 for i in range(n_bins)]
    bin_conf = [bin_conf[i] / bin_size[i] if bin_size[i] != 0 else 0.0 for i in range(n_bins)]
    for i in range(n_bins):
        if bin_size[i] == 0:
            continue

        ECE += abs(bin_acc[i] - bin_conf[i]) * bin_size[i] / len(y_true)

    return ECE