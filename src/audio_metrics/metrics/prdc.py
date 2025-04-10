import torch


def nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: torch.Tensor([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = torch.cdist(input_features, input_features)
    radii = torch.kthvalue(distances, k=nearest_k + 1, dim=-1)[0]
    return radii


# adapted from https://github.com/clovaai/generative-evaluation-prdc/pull/10
def prdc(reference, candidate, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        reference: torch.Tensor([N, feature_dim], dtype=torch.float32)
        candidate: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    # print("Num real: {} Num fake: {}".format(reference.shape[0], candidate.shape[0]))
    ref_nn_radii = reference.get_radii(nearest_k)
    cand_nn_radii = candidate.get_radii(nearest_k)

    distance_ref_cand = torch.cdist(reference.embeddings, candidate.embeddings)

    precision = (
        (distance_ref_cand < ref_nn_radii[:, None]).any(dim=0).double().mean().item()
    )

    recall = (
        (distance_ref_cand < cand_nn_radii[None, :]).any(dim=1).double().mean().item()
    )

    density = (1.0 / float(nearest_k)) * (distance_ref_cand < ref_nn_radii[:, None]).sum(
        dim=0
    ).double().mean().item()

    coverage = (distance_ref_cand.min(dim=1)[0] < ref_nn_radii).double().mean().item()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage)
