from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

import laion_clap

from audio_metrics.dataset import Embedder, load_audio

CHECKPOINT = "music_audioset_epoch_15_esc_90.14.pt"
SR = 48000


class CLAP(Embedder):
    def __init__(self, device):
        super().__init__()
        self.model = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base"
        ).to(device)
        self.model.load_ckpt(CHECKPOINT)
        self.sr = SR
        self.activations = defaultdict(list)
        self.layers = ["audio_projection.0", "audio_projection.2"]
        for layer in self.layers:
            self.model.get_submodule(f"model.{layer}").register_forward_hook(
                self._get_activation_hook(layer)
            )

    def preprocess_path(self, fp):
        # load audio as numpy array from file
        audio, _ = load_audio(fp, self.sr, mono=True, dtype=np.float32)
        audio = audio.reshape((1, -1))
        return audio

    def embed(self, dataset):
        self._clear_activations()
        # todo: batch data using DataLoader (set use_tensor=True in get_audio_embedding_from_data)
        for audio in dataset:
            self.model.get_audio_embedding_from_data(x=audio.reshape((1, -1)))
        result = {
            key: torch.cat(acts, 0).cpu().numpy() for key, acts in self.activations.items()
        }
        self._clear_activations()
        return result

    def _clear_activations(self):
        self.activations = defaultdict(list)

    def _get_activation_hook(self, name):
        def hook(model, input, output):
            # need clone?
            self.activations[name].append(output.detach())
        return hook

