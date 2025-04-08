import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA as IncrementalPCA_orig


class IncrementalPCA(IncrementalPCA_orig):
    conversions = {
        "array": [
            "components_",
            "mean_",
            "var_",
            "singular_values_",
            "explained_variance_",
            "explained_variance_ratio_",
        ],
        "int64": ["n_samples_seen_"],
        "float64": ["noise_variance_"],
    }

    def transform(self, x):
        return torch.as_tensor(super().transform(x))

    def __getstate__(self):
        state = super().__getstate__().copy()
        for k in self.conversions["array"]:
            if k in state:
                state[k] = torch.as_tensor(state[k])
        for k in self.conversions["int64"]:
            if k in state:
                state[k] = int(state[k])
        for k in self.conversions["float64"]:
            if k in state:
                state[k] = float(state[k])
        return state

    def __setstate__(self, state):
        for k in self.conversions["array"]:
            if k in state:
                state[k] = state[k].numpy()
        for k in self.conversions["int64"]:
            if k in state:
                state[k] = np.int64(state[k])
        for k in self.conversions["float64"]:
            if k in state:
                state[k] = np.float64(state[k])
        super().__setstate__(state)


# fp = "/tmp/out.pt"
# x = np.random.random((100, 30))

# pca = IncrementalPCA(n_components=10)
# pca.partial_fit(x)
# y = pca.transform(x)

# state = pca.__getstate__()
# torch.save(state, fp)
# new_state = torch.load(fp, weights_only=True)

# new_pca = IncrementalPCA()
# new_pca.__setstate__(new_state)
# yp = new_pca.transform(x)

# print(np.all(y == yp))
# print(np.max(np.abs(y - yp)))
# # print(new_state)
