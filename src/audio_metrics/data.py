import torch
import prdc


def ensure_tensor(x, device=None):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(device, non_blocking=True) if device else x


class AudioMetricsData:
    def __init__(self, store_embeddings=True):
        self.mean = None
        self.n = None
        self.cov = None
        self.store_embeddings = store_embeddings
        self.embeddings = None
        self.radii = {}
        self.dtype = torch.float64

    def save(self, fp):
        torch.save(self.__dict__, fp)

    @classmethod
    def load(cls, fp):
        self = cls()
        self.__dict__.update(torch.load(fp, weights_only=True))
        return self

    def add(self, embeddings):
        mean = torch.mean(embeddings, 0).to(dtype=self.dtype)
        cov = torch.cov(embeddings.T).to(dtype=self.dtype)
        n = len(embeddings)
        self._update_stats(mean, cov, n)
        if self.store_embeddings:
            self._update_embeddings(embeddings)

    def get_radii(self, k_neighbor):
        key = f"radii_{k_neighbor}"
        radii = self.radii.get(key)
        if radii is None and self.embeddings is not None:
            radii = prdc.compute_nearest_neighbour_distances(
                self.embeddings.numpy(), k_neighbor
            )
            self.radii[key] = radii
        return radii

    def _update_embeddings(self, embeddings):
        if self.embeddings is None:
            self.embeddings = embeddings
            return
        self.embeddings = torch.cat((self.embeddings, embeddings))

    def __len__(self):
        return self.n or 0

    def _update_stats(self, mean, cov, n):
        if self.n is None:
            self.mean = mean
            self.cov = cov
            self.n = n
            return
        n_prod = self.n * n
        n_total = self.n + n
        new_mean = (self.n * self.mean + n * mean) / n_total
        diff_mean = self.mean - mean
        diff_mean_mat = torch.einsum("i,j->ij", diff_mean, diff_mean)
        w_self = (self.n - 1) / (n_total - 1)
        w_other = (n - 1) / (n_total - 1)
        w_diff = (n_prod / n_total) / (n_total - 1)
        new_cov = w_self * self.cov + w_other * cov + w_diff * diff_mean_mat
        self.n = n_total
        self.mean = new_mean
        self.cov = new_cov

    def __iadd__(self, other):
        assert isinstance(other, AudioMetricsData)
        if self.n is None:
            self.store_embeddings = other.store_embeddings
        assert self.store_embeddings == other.store_embeddings
        assert other.n is not None
        self._update_stats(other.mean, other.cov, other.n)
        if self.store_embeddings:
            self._update_embeddings(other.embeddings)

    def __add__(self, other):
        new = AudioMetricsData()
        new += other
        return new
