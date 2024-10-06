import pytest
import pickle
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    AccompanimentPromptAdherence,
)
from audio_metrics.example_utils import generate_audio_samples


@pytest.fixture
def samples_dir(tmpdir):
    audio_dir = tmpdir / "samples"
    generate_audio_samples(audio_dir, n_items=20)
    return audio_dir


def make_apa(win_dur=5.0, dev="cuda"):
    return AccompanimentPromptAdherence(
        torch.device(dev), win_dur, n_pca=10, embedder="openl3", metric="fad"
    )


def make_std_apa(dev="cuda"):
    return AudioPromptAdherence(
        torch.device(dev),
        **{
            "embedder": "clap_music_speech",
            "win_dur": 5.0,
            "layer": "audio_projection.2",
            "metric": "fad",
            "n_pca": None,
            "pca_whiten": False,
            "mix_func": "L0",
            "_state": "/tmp/mg/apa_bg_sapa.npz",
        }
    )


@pytest.fixture
def apa():
    return make_apa()


def trim_state(state_fp):
    with open(state_fp, "rb") as f:
        state = pickle.load(f)

    n_1 = len(state["metrics_1/bg_data"][("emb", "output")].activations)
    n_2 = len(state["metrics_2/bg_data"][("emb", "output")].activations)

    state["metrics_1/bg_data"][("emb", "output")].activations = DummyActivations(n_1)
    state["metrics_2/bg_data"][("emb", "output")].activations = DummyActivations(n_2)
    state["metrics_1/_pca_bg_data"][("emb", "output")].activations = DummyActivations(n_1)
    state["metrics_2/_pca_bg_data"][("emb", "output")].activations = DummyActivations(n_2)

    with open(state_fp, "wb") as f:
        pickle.dump(state, f)


class DummyActivations:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_apa_save_state(samples_dir, apa, tmpdir):
    # load audio samples from files in `audio_dir`
    real_items = async_audio_loader(samples_dir / "real", mono=False)
    fake_items = async_audio_loader(samples_dir / "fake", mono=False)
    # iterate over windows
    real_items = multi_audio_slicer(real_items, apa.win_dur)
    fake_items = multi_audio_slicer(fake_items, apa.win_dur)

    # apa.set_background(real_items)
    # result1 = apa.compare_to_background(fake_items)
    # # state_fp = tmpdir / "apa_state.pkl"
    # # apa.save_state(state_fp)

    state_fp = "/tmp/mg/apa_bg_sapa.npz"
    # trim_state(state_fp)

    print("a")
    apa1 = make_std_apa()
    # apa1.load_state(state_fp)

    fake_items = async_audio_loader(samples_dir / "fake", mono=False)
    fake_items = multi_audio_slicer(fake_items, apa1.win_dur)

    print("b")
    result2 = apa1.compare_to_background(fake_items)
    print(result2)
    # assert result1 == pytest.approx(result2, rel=1e-4, abs=1e-4)
