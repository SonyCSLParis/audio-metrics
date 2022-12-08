import numpy as np
from prdc import prdc


def compute_density_coverage(real, fake, nearest_k):
    distance_real_fake = prdc.compute_pairwise_distance(
        real.activations, fake.activations
    )
    real_radii = real.get_radii(nearest_k)
    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_radii, axis=1)
    ).sum(axis=0).mean()
    coverage = (distance_real_fake.min(axis=1) < real_radii).mean()
    return density, coverage
