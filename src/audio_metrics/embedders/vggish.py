import torch
import einops


class VGGish:
    def __init__(self):
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.model.eval()
        self.model = self.model.to("cpu")
        self.model.postprocess = False
        self.model.preprocess = False
        self.model.embeddings[5] = torch.nn.Identity()

    @property
    def sr(self):
        return 16000

    def get_device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, data, sr=None):
        device = self.get_device()
        audio = data["audio"]
        self.model.device = device
        batch_size = len(audio)
        spec = [self.model._preprocess(item, self.sr) for item in audio]
        spec = einops.rearrange(spec, "b t 1 w h -> (b t) 1 w h")
        spec = spec.to(device)
        embedding = self.model(spec)
        embedding = einops.rearrange(embedding, "(b t) d -> b t d", b=batch_size)
        embedding = embedding.mean(1)
        return {"embedding": embedding}
