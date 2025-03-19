from pathlib import Path
from functools import partial
from itertools import chain
from time import perf_counter

import numpy as np
import torch
import laion_clap

from audio_metrics.example_utils import generate_audio_samples
from audio_metrics.mix_functions import mix_pair, MIX_FUNCTIONS
from audio_metrics.cpu_parallel import iterable_process as cpu_iterable_process
from audio_metrics.gpu_parallel import iterable_process as gpu_iterable_process


from audio_metrics import async_audio_loader, multi_audio_slicer, APA


class CLAP:
    def __init__(self, ckpt, model_name="clap"):
        self.model_name = model_name
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.ckpt_path = Path(ckpt)
        self.clap.load_ckpt(ckpt, verbose=False)
        self.sr = 48000  # hard-coded for this laion_clap.CLAP_Module

    def get_device(self):
        return next(self.clap.parameters()).device

    @torch.no_grad()
    def forward(self, audio, sr):
        assert sr == self.sr
        audio = torch.from_numpy(audio).float().to(self.get_device(), non_blocking=True)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        embedding = self.clap.get_audio_embedding_from_data(
                audio, use_tensor=True
            )
        return embedding.cpu()



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

def batch_iterator(items, batch_size = 32):
    batch = []
    for audio, sr in items:
        batch.append(audio)
        if len(batch) == batch_size:
            yield np.stack(batch), sr
            batch = []
    if batch:
        yield np.stack(batch), sr

# def unbatch_iterator(items):
#     for batch in items:
#         for item in batch:
#             yield item

    
def main():
    win_dur = 5.0
    (
        real1_win_iterator,
        fake1_win_iterator,
        real2_win_iterator,
    ) = get_data_iterators(win_dur)
    clap_encoder = CLAP(ckpt='/home/maarten/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt')

    mix_func = partial(mix_pair, mix_func=MIX_FUNCTIONS["L0"])

    items = ((item,) for item in chain(real1_win_iterator, fake1_win_iterator, real2_win_iterator))

    items = cpu_iterable_process(
        items,
        mix_func,
        n_workers=128,
        desc="mixing pairs",
    )
    items = batch_iterator(items, batch_size=32)
    items = gpu_iterable_process(
        items,
        clap_encoder,
        desc="computing clap",
        n_gpus=2,
        prefetch_factor=8
    )
    t0  = perf_counter()
    activations = torch.cat(list(items))
    t1  = perf_counter()
    print(f'dur: {t1-t0:.3f}s')
    print(activations.shape)
    # import ipdb

    # ipdb.set_trace()
    # apa = APA()
    # reference_set = load_ctx_stem_pairs()
    # # candidate_set = ...
    # apa.set_reference(reference_set)
    # apa.compute(candidate_set)


if __name__ == "__main__":
    main()
