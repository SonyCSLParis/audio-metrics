# From https://github.com/toshas/torch-fidelity/tree/master/torch_fidelity

# Functions mmd2 and polynomial_kernel are adapted from
#   https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
#   Distributed under BSD 3-Clause: https://github.com/mbinkowski/MMD-GAN/blob/master/LICENSE

import numpy as np
import torch
from tqdm import tqdm
import logging

KEY_METRIC_KID_MEAN = "kernel_distance_mean"
KEY_METRIC_KID_STD = "kernel_distance_std"
KID_SUBSETS = 100
KID_SUBSET_SIZE = 1000
KID_DEGREE = 3
KID_GAMMA = None
KID_COEF0 = 1


def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est="unbiased"):
    assert mmd_est in (
        "biased",
        "unbiased",
        "u-statistic",
    ), "Invalid value of mmd_est"

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == "biased":
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == "unbiased":
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    return mmd2


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    k_11 = polynomial_kernel(
        features_1, features_1, degree=degree, gamma=gamma, coef0=coef0
    )
    k_22 = polynomial_kernel(
        features_2, features_2, degree=degree, gamma=gamma, coef0=coef0
    )
    k_12 = polynomial_kernel(
        features_1, features_2, degree=degree, gamma=gamma, coef0=coef0
    )
    return mmd2(k_11, k_12, k_22)


def kid_features_to_metric(features_1, features_2, **kwargs):
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]

    # kid_subsets = get_kwarg("kid_subsets", kwargs)
    # kid_subset_size = get_kwarg("kid_subset_size", kwargs)
    # verbose = get_kwarg("verbose", kwargs)
    kid_subsets = kwargs.get("kid_subsets", KID_SUBSETS)
    kid_subset_size = kwargs.get("kid_subset_size", KID_SUBSET_SIZE)
    verbose = kwargs.get("verbose", False)

    n_samples_1, n_samples_2 = len(features_1), len(features_2)
    assert (
        n_samples_2 and n_samples_2
    ), "Cannot compute KID on empty features tensor"
    n_samples = min(n_samples_1, n_samples_2)
    if kid_subset_size >= n_samples:
        new_ss = max(1, n_samples // 2)
        if verbose:
            msg = (
                f"Reducing KID subset size from {kid_subset_size} to {new_ss} "
                "to accommodate small sample size"
            )
            logging.warning(msg)
        kid_subset_size = new_ss

    # assert n_samples_1 >= kid_subset_size and n_samples_2 >= kid_subset_size, (
    #     f"KID subset size {kid_subset_size} cannot be smaller than the number of samples (input_1: {n_samples_1}, "
    #     f'input_2: {n_samples_2}). Consider using "kid_subset_size" kwarg or "--kid-subset-size" command line key to '
    #     f"proceed."
    # )

    features_1 = features_1.cpu().numpy()
    features_2 = features_2.cpu().numpy()

    mmds = np.zeros(kid_subsets)
    rng = np.random.default_rng(kwargs.get("rng_seed", 1234))

    for i in tqdm(
        range(kid_subsets),
        disable=not verbose,
        leave=False,
        unit="subsets",
        desc="Kernel Inception Distance",
    ):
        f1 = features_1[rng.choice(n_samples_1, kid_subset_size, replace=False)]
        f2 = features_2[rng.choice(n_samples_2, kid_subset_size, replace=False)]
        o = polynomial_mmd(
            f1,
            f2,
            kwargs.get("kid_degree", KID_DEGREE),
            kwargs.get("kid_gamma", KID_GAMMA),
            kwargs.get("kid_coef0", KID_COEF0),
        )
        mmds[i] = o

    out = {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }

    return out


compute_kernel_distance = kid_features_to_metric


# def kid_featuresdict_to_metric(
#     featuresdict_1, featuresdict_2, feat_layer_name, **kwargs
# ):
#     features_1 = featuresdict_1[feat_layer_name]
#     features_2 = featuresdict_2[feat_layer_name]

#     # make sur
#     # kid_subsets = kwargs.get("kid_subsets", 100)
#     # kid_subset_size = kwargs.get("kid_subset_size", 1000)

#     metric = kid_features_to_metric(features_1, features_2, **kwargs)
#     return metric


if __name__ == "__main__":
    emb = 128
    bs = 800
    shape = (bs, emb)
    f1 = torch.randn(shape)
    f2 = torch.randn(shape)
    print(kid_features_to_metric(f1, f2))
