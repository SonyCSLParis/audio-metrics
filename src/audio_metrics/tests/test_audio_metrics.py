import pytest
import numpy as np
import torch
from audio_metrics import AudioMetrics


class DummyEmbedder:
    def __init__(self):
        self.m = torch.nn.Linear(1, 1)

    @property
    def sr(self):
        return 16000

    def get_device(self):
        return next(self.m.parameters()).device

    @torch.no_grad()
    def forward(self, data, sr=None):
        # create some embedding that is deterministic given the the data but not constant
        # data: {'audio': np.array(...)} # audio shape: [batch_size, n_samples]
        mean = torch.as_tensor(10**3 * data["audio"].std(axis=1))
        embedding = torch.outer(mean, torch.arange(10))
        return {"embedding": embedding}


def mix_func(audio, sr=None):
    return audio.mean(axis=1)


def test_serialization():
    kwargs = dict(
        embedder=DummyEmbedder(),
        mix_function=mix_func,
        metrics=["fad", "apa"],
        n_pca=10,
    )
    am = AudioMetrics(**kwargs)
    sr = 16000
    n_seconds = 5
    reference = (np.random.random((n_seconds * sr, 2)) for _ in range(100))
    candidate_good = [np.random.random((n_seconds * sr, 2)) for _ in range(100)]
    am.add_reference(reference)
    result1 = am.evaluate(candidate_good)
    out_fp = "/tmp/out.pt"
    am.save_state(out_fp)
    am = AudioMetrics(**kwargs)
    am.load_state(out_fp)
    result2 = am.evaluate(candidate_good)
    assert len(result1) == len(result2)
    for k, v1 in result1.items():
        v2 = result2[k]
        assert v1 == pytest.approx(v2, rel=1e-6, abs=1e-6)
