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


@pytest.fixture
def audio_metrics_instance():
    kwargs = dict(
        embedder=DummyEmbedder(),
        mix_function=mix_func,
        metrics=["fad", "apa"],
        n_pca=10,
    )
    am = AudioMetrics(**kwargs)
    return am


@pytest.fixture
def inputs_1():
    sr = 16000
    n_seconds = 5
    reference = np.random.random((100, n_seconds * sr, 2))
    candidate = np.random.random((100, n_seconds * sr, 2))
    return reference, candidate


@pytest.fixture
def inputs_2():
    sr = 16000
    n_seconds = 5
    reference = (np.random.random((n_seconds * sr, 2)) for _ in range(100))
    candidate = [np.random.random((n_seconds * sr, 2)) for _ in range(100)]
    return reference, candidate


@pytest.fixture
def inputs_3():
    sr = 16000
    n_seconds = 5
    reference = torch.randn((100, n_seconds * sr, 2))
    candidate = torch.randn((100, n_seconds * sr, 2))
    return reference, candidate


# @pytest.fixture
# def inputs_a():
#     sr = 48000
#     n_seconds = 5
#     reference = (
#         {"audio": np.random.random((n_seconds * sr, 2)), "sr": sr} for _ in range(100)
#     )
#     candidate = (
#         {"audio": np.random.random((n_seconds * sr, 2)), "sr": sr} for _ in range(100)
#     )

#     return reference, candidate


# @pytest.fixture
# def inputs_b():
#     sr = 48000
#     n_seconds = 5
#     reference = (
#         {"audio": torch.randn((n_seconds * sr, 2)), "sr": sr} for _ in range(100)
#     )
#     candidate = (
#         {"audio": torch.randn((n_seconds * sr, 2)), "sr": sr} for _ in range(100)
#     )

#     return reference, candidate


# def test_inputs_a(audio_metrics_instance, inputs_a):
#     am = audio_metrics_instance
#     am.reset_reference()
#     reference, candidate = inputs_a
#     am.add_reference(reference)
#     am.evaluate(candidate)


# def test_inputs_b(audio_metrics_instance, inputs_b):
#     am = audio_metrics_instance
#     am.reset_reference()
#     reference, candidate = inputs_b
#     am.add_reference(reference)
#     am.evaluate(candidate)


def test_inputs_1(audio_metrics_instance, inputs_1):
    am = audio_metrics_instance
    am.reset_reference()
    reference, candidate = inputs_1
    am.add_reference(reference)
    am.evaluate(candidate)


def test_inputs_2(audio_metrics_instance, inputs_2):
    am = audio_metrics_instance
    am.reset_reference()
    reference, candidate = inputs_2
    am.add_reference(reference)
    am.evaluate(candidate)


def test_inputs_3(audio_metrics_instance, inputs_3):
    am = audio_metrics_instance
    am.reset_reference()
    reference, candidate = inputs_3
    am.add_reference(reference)
    am.evaluate(candidate)


@pytest.fixture
def audio_metrics_instance_no_apa():
    kwargs = dict(
        embedder=DummyEmbedder(),
        mix_function=mix_func,
        metrics=["fad"],
        n_pca=10,
    )
    am = AudioMetrics(**kwargs)
    return am


@pytest.fixture
def inputs_stems_1():
    sr = 16000
    n_seconds = 5
    reference = np.random.random((100, n_seconds * sr))
    candidate = np.random.random((100, n_seconds * sr))
    return reference, candidate


def test_inputs_4(audio_metrics_instance_no_apa, inputs_stems_1):
    am = audio_metrics_instance_no_apa
    am.reset_reference()
    reference, candidate = inputs_stems_1
    am.add_reference(reference)
    am.evaluate(candidate)


def test_inputs_5(audio_metrics_instance, inputs_stems_1):
    am = audio_metrics_instance
    am.reset_reference()
    reference, candidate = inputs_stems_1
    with pytest.raises(ValueError):
        am.add_reference(reference)


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
