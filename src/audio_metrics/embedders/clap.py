import torch
from ..util.get_url import get_file

CLAP_MUSIC_CHECKPOINT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"


class CLAP:
    def __init__(self, ckpt=None, model_name="clap"):
        import laion_clap

        self.model_name = model_name
        if ckpt is None:
            ckpt = get_file(CLAP_MUSIC_CHECKPOINT_URL)
            print(ckpt)
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(ckpt, verbose=False)

    @property
    def sr(self):
        return self.clap.model.audio_cfg.sample_rate  # 48000

    def get_device(self):
        return next(self.clap.parameters()).device

    @torch.no_grad()
    def forward(self, data, sr=None):
        audio = (
            torch.from_numpy(data["audio"])
            .float()
            .to(self.get_device(), non_blocking=True)
        )
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        embedding = self.clap.get_audio_embedding_from_data(audio, use_tensor=True)
        return {"embedding": embedding}

    # todo: adapt
    def _register_hooks(self, act_dict):
        hooks = []
        for layer in self.layers:
            hooks.append(
                self.clap.get_submodule(f"model.{layer}").register_forward_hook(
                    self._get_activation_hook(layer, act_dict)
                )
            )
        return hooks

    def _clear_activations(self):
        self.activations = defaultdict(list)

    def _get_activation_hook(self, name, act_dict):
        def hook(model, input, output):
            act_dict[name] = output.cpu().unsqueeze(1).numpy()

        return hook
