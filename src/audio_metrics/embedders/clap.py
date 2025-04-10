from functools import partial
import torch
from ..util.get_url import download_url

LAION_CLAP_MUSIC_SPEECH_CHECKPOINT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt"
LAION_CLAP_MUSIC_CHECKPOINT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
LAION_CLAP_LAYERS = ["audio_projection.0", "audio_projection.2"]


class LaionCLAP:
    def __init__(self, ckpt=None, layer=None):
        import laion_clap

        if ckpt is None:
            ckpt = LAION_CLAP_MUSIC_CHECKPOINT_URL
        ckpt = download_url(ckpt)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(ckpt, verbose=False)
        self.layer = layer

    @property
    def sr(self):
        return self.clap.model.audio_cfg.sample_rate  # 48000

    def get_device(self):
        return next(self.clap.parameters()).device

    @torch.no_grad()
    def forward(self, data, sr=None):
        audio = self._prepare_audio(data)
        if self.layer:
            # inefficient to do this at each forward, but doing it globally may
            # not be thread/process safe? Check
            layer_output = {}
            hook = self._register_hook(
                partial(activation_hook, storage=layer_output, name=self.layer)
            )

        embedding = self.clap.get_audio_embedding_from_data(audio, use_tensor=True)

        if self.layer:
            hook.remove()
            embedding = layer_output[self.layer]
        return {"embedding": embedding}

    def _prepare_audio(self, data):
        audio = (
            torch.from_numpy(data["audio"])
            .float()
            .to(self.get_device(), non_blocking=True)
        )
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        return audio

    def _register_hook(self, hook_func):
        hook = self.clap.get_submodule(f"model.{self.layer}").register_forward_hook(
            hook_func
        )
        return hook


def activation_hook(model, input, output, storage, name):
    storage[name] = output


CLAP = LaionCLAP
