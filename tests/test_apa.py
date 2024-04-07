import pytest
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    AudioPromptAdherence,
)
from audio_metrics.example_utils import generate_audio_samples


@pytest.fixture
def samples_dir(tmpdir):
    audio_dir = tmpdir / "samples"
    generate_audio_samples(audio_dir, n_items=20)
    return audio_dir


@pytest.fixture
def apa():
    win_dur = 5.0
    dev = torch.device("cuda")
    return AudioPromptAdherence(dev, win_dur, n_pca=10, embedder="openl3", metric="fad")


def test_apa_save_state(samples_dir, apa, tmpdir):
    # load audio samples from files in `audio_dir`
    real_items = async_audio_loader(samples_dir / "real", mono=False)
    fake_items = async_audio_loader(samples_dir / "fake", mono=False)
    # iterate over windows
    real_items = multi_audio_slicer(real_items, apa.win_dur)
    fake_items = multi_audio_slicer(fake_items, apa.win_dur)

    apa.set_background(real_items)
    result1 = apa.compare_to_background(fake_items)

    state_fp = tmpdir / "apa_state.npz"
    apa.save_state(state_fp)
    apa.load_state(state_fp)

    fake_items = async_audio_loader(samples_dir / "fake", mono=False)
    fake_items = multi_audio_slicer(fake_items, apa.win_dur)
    result2 = apa.compare_to_background(fake_items)

    assert result1 == result2
