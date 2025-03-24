from pathlib import Path
import torch
from benedict import benedict

# from functools import partial
# import random
# from itertools import chain, tee

from audio_metrics.embed import embedding_pipeline, ItemCategory
from audio_metrics.metrics.fad import frechet_distance
from audio_metrics.metrics.apa import apa
from audio_metrics.embedders.clap import CLAP
from audio_metrics.example_utils import generate_audio_samples
from audio_metrics.util.gpu_parallel import GPUWorkerHandler

from audio_metrics.old.dataset import async_audio_loader


def get_data_iterators(
    basedir=".",
    n_items=500,
    sr=48000,
):
    audio_dir1 = Path(basedir) / "audio_samples1"
    audio_dir2 = Path(basedir) / "audio_samples2"

    # print("generating 'real' and 'fake' audio samples")
    if not audio_dir1.exists():
        generate_audio_samples(audio_dir1, sr=sr, n_items=n_items)
    if not audio_dir2.exists():
        generate_audio_samples(audio_dir2, sr=sr, n_items=n_items)

    # load audio samples from files in `audio_dir`
    real1_song_iterator = async_audio_loader(audio_dir1 / "real", mono=False)
    fake1_song_iterator = async_audio_loader(audio_dir1 / "fake", mono=False)
    real2_song_iterator = async_audio_loader(audio_dir2 / "real", mono=False)

    real1_song_iterator = (item for item, _ in real1_song_iterator)
    fake1_song_iterator = (item for item, _ in fake1_song_iterator)
    real2_song_iterator = (item for item, _ in real2_song_iterator)

    return (
        real1_song_iterator,
        fake1_song_iterator,
        real2_song_iterator,
    )


def main():
    clap_cktpt = "/home/maarten/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt"
    clap_encoder = CLAP(ckpt=clap_cktpt)
    # clap_encoder = benedict({"sr": 48000})
    (
        real1_song_iterator,
        fake1_song_iterator,
        real2_song_iterator,
    ) = get_data_iterators(sr=clap_encoder.sr)
    # waveforms = chain(real1_song_iterator, fake1_song_iterator, real2_song_iterator)
    n_gpus = torch.cuda.device_count()
    gpu_handler = GPUWorkerHandler(n_gpus)

    stems_mode = True

    # if stems_mode and apa_mode is None:
    #     real2_song_iterator = (item[:, 0] for item in real2_song_iterator)

    reference_metrics = embedding_pipeline(
        real2_song_iterator,
        clap_encoder,
        apa_mode="reference",
        stems_mode=stems_mode,
        n_gpus=n_gpus,
        gpu_handler=gpu_handler,
    )

    cand_real_metrics = embedding_pipeline(
        real1_song_iterator,
        clap_encoder,
        apa_mode="candidate",
        stems_mode=stems_mode,
        n_gpus=n_gpus,
        gpu_handler=gpu_handler,
    )
    cand_fake_metrics = embedding_pipeline(
        fake1_song_iterator,
        clap_encoder,
        apa_mode="candidate",
        stems_mode=stems_mode,
        n_gpus=n_gpus,
        gpu_handler=gpu_handler,
    )

    rrp = frechet_distance(
        reference_metrics[ItemCategory.aligned],
        reference_metrics[ItemCategory.misaligned],
    )
    cr = frechet_distance(
        cand_real_metrics[ItemCategory.aligned], reference_metrics[ItemCategory.aligned]
    )
    crp = frechet_distance(
        cand_real_metrics[ItemCategory.aligned],
        reference_metrics[ItemCategory.misaligned],
    )
    cpr = frechet_distance(
        cand_fake_metrics[ItemCategory.aligned], reference_metrics[ItemCategory.aligned]
    )
    cprp = frechet_distance(
        cand_fake_metrics[ItemCategory.aligned],
        reference_metrics[ItemCategory.misaligned],
    )
    print("real")
    print(apa(cr, crp, rrp))
    print("fake")
    print(apa(cpr, cprp, rrp))
    if stems_mode:
        fad_real = frechet_distance(
            reference_metrics[ItemCategory.stem],
            cand_real_metrics[ItemCategory.stem],
        )
        fad_fake = frechet_distance(
            reference_metrics[ItemCategory.stem],
            cand_fake_metrics[ItemCategory.stem],
        )
        print("fad real", fad_real)

        print("fad fake", fad_fake)

    import ipdb

    ipdb.set_trace()
    reference_metrics
    # C = embedding_pipeline_single(
    #     real1_song_iterator,
    #     clap_encoder,
    #     n_gpus=n_gpus,  # gpu_handler=gpu_handler
    # )
    # Cp = embedding_pipeline_single(
    #     fake1_song_iterator,
    #     clap_encoder,
    #     n_gpus=n_gpus,  # gpu_handler=gpu_handler
    # )


if __name__ == "__main__":
    main()
