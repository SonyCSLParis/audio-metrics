import torch


class CLAP:
    def __init__(self, ckpt, model_name="clap"):
        import laion_clap

        self.model_name = model_name
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
