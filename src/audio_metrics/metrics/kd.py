# From https://github.com/toshas/torch-fidelity/tree/master/torch_fidelity

# Functions mmd2 and polynomial_kernel are adapted from
#   https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
#   Distributed under BSD 3-Clause: https://github.com/mbinkowski/MMD-GAN/blob/master/LICENSE

from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm
import logging

from audio_metrics.data import AudioMetricsData, ensure_ndarray

KEY_METRIC_KID_MEAN = "kernel_distance_mean"
KEY_METRIC_KID_STD = "kernel_distance_std"
KID_SUBSETS = 100
KID_SUBSET_SIZE = 1000
# Polynomial kernel
KID_DEGREE = 3
KID_GAMMA = None
KID_COEF0 = 1
# RBF kernel
KID_SIGMA = 10.0


def kernel_distance(
    x: AudioMetricsData,
    y: AudioMetricsData,
):
    return kid_features_to_metric(
        ensure_ndarray(x.embeddings), ensure_ndarray(y.embeddings)
    )


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


def rbf_kernel(X, Y, sigma=1.0):
    """
    Compute the RBF (Gaussian) kernel between X and Y using scipy's cdist for
    efficient distance computation.

    Parameters:

        - X: numpy array of shape (n_samples_X, n_features)

        - Y: numpy array of shape (n_samples_Y, n_features)

        - sigma: float, the width parameter of the RBF kernel.

    Returns:

        - kernel_matrix: numpy array of shape (n_samples_X, n_samples_Y)
    """
    # Compute the squared Euclidean distance using cdist
    squared_dist = cdist(X, Y, "sqeuclidean")

    # Compute the RBF kernel matrix
    kernel_matrix = np.exp(-squared_dist / (2 * sigma**2))

    return kernel_matrix


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def kernel_mmd2(features_1, features_2, kernel):
    k_11 = kernel(features_1, features_1)
    k_22 = kernel(features_2, features_2)
    k_12 = kernel(features_1, features_2)
    # return max(0, mmd2(k_11, k_12, k_22, mmd_est="unbiased")) ** 0.5
    return mmd2(k_11, k_12, k_22, mmd_est="unbiased")


def kid_features_to_metric(features_1, features_2, **kwargs):
    kernel_type = kwargs.get("kernel_type", "polynomial")
    if kernel_type == "polynomial":
        kernel = partial(
            polynomial_kernel,
            degree=kwargs.get("kid_degree", KID_DEGREE),
            gamma=kwargs.get("kid_gamma", KID_GAMMA),
            coef0=kwargs.get("kid_coef0", KID_COEF0),
        )
    elif kernel_type == "rbf":
        kernel = partial(
            rbf_kernel,
            sigma=kwargs.get("kid_sigma", KID_SIGMA),
        )
    else:
        raise NotImplementedError(f'Unknown kernel_type "{kernel_type}"')

    if torch.is_tensor(features_1):
        features_1 = features_1.cpu().numpy()
    if torch.is_tensor(features_2):
        features_2 = features_2.cpu().numpy()

    assert features_1.ndim == 2
    assert features_2.ndim == 2
    assert features_1.shape[1] == features_2.shape[1]

    kid_subsets = kwargs.get("kid_subsets", KID_SUBSETS)
    kid_subset_size = kwargs.get("kid_subset_size", KID_SUBSET_SIZE)
    verbose = kwargs.get("verbose", False)

    n_samples_1, n_samples_2 = len(features_1), len(features_2)
    assert n_samples_2 and n_samples_2, "Cannot compute KID on empty features tensor"
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
        mmds[i] = kernel_mmd2(f1, f2, kernel)

    out = {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }

    return out


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
