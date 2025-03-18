from pathlib import Path
import json
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    AccompanimentPromptAdherence,
)
from audio_metrics.example_utils import generate_audio_samples


def get_data_iterators(win_dur, basedir="."):
    audio_dir1 = Path(basedir) / "audio_samples1"
    audio_dir2 = Path(basedir) / "audio_samples2"

    # print("generating 'real' and 'fake' audio samples")
    if not audio_dir1.exists():
        generate_audio_samples(audio_dir1)
    if not audio_dir2.exists():
        generate_audio_samples(audio_dir2)

    # load audio samples from files in `audio_dir`
    real1_song_iterator = async_audio_loader(audio_dir1 / "real", mono=False)
    fake1_song_iterator = async_audio_loader(audio_dir1 / "fake", mono=False)
    real2_song_iterator = async_audio_loader(audio_dir2 / "real", mono=False)

    # iterate over windows
    real1_win_iterator = multi_audio_slicer(real1_song_iterator, win_dur)
    fake1_win_iterator = multi_audio_slicer(fake1_song_iterator, win_dur)
    real2_win_iterator = multi_audio_slicer(real2_song_iterator, win_dur)

    return (
        real1_win_iterator,
        fake1_win_iterator,
        real2_win_iterator,
    )


def main():
    win_dur = 5.0
    (
        real1_win_iterator,
        fake1_win_iterator,
        real2_win_iterator,
    ) = get_data_iterators(win_dur)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = AccompanimentPromptAdherence(
        dev,
        win_dur,
        n_pca=None,
        embedder="clap_music",
        metric="fad",
        layer="audio_projection.2",
    )
    metrics.set_background(real1_win_iterator)
    print("non-matching:")
    result = metrics.compare_to_background(fake1_win_iterator)
    print(json.dumps(result, indent=2))
    print("matching:")
    result = metrics.compare_to_background(real2_win_iterator)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
