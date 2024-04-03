# from pathlib import Path
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

import torchopenl3

from audio_metrics.dataset import Embedder, load_audio, GeneratorDataset


class OpenL3(Embedder):
    def __init__(self, device):
        super().__init__(sr=torchopenl3.core.TARGET_SR, mono=True)
        self.model = torchopenl3.core.load_audio_embedding_model(
            "mel256", "music", 6144
        )
        self.device = device
        self.hop_size = 1.0
        self.activations = defaultdict(list)
        self.out_label = "output"
        self.names = [self.out_label]

    def embed(self, items, same_size=False, batch_size=10):
        audio_sr_pairs = (self.preprocess(item) for item in items)
        num_workers = 0
        if same_size:
            dataset = GeneratorDataset(
                (audio for audio, _ in audio_sr_pairs),
                multi=False,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            dataloader = (
                torch.from_numpy(audio) for audio, _ in audio_sr_pairs
            )
        pred_arr = []
        with torch.no_grad():
            for batch in dataloader:
                audio_emb, _ = torchopenl3.get_audio_embedding(
                    # batch.to(self.device).unsqueeze(1),
                    batch.to(self.device),
                    self.sr,
                    self.model,
                    center=False,
                    hop_size=self.hop_size,
                )
                if not same_size:
                    assert audio_emb.size(0) == 1
                    audio_emb = audio_emb.squeeze(0)
                pred_arr.append(audio_emb.cpu().numpy())

        if same_size:
            # pred_arr = torch.cat(pred_arr).cpu().numpy()
            pred_arr = np.concatenate(pred_arr, 0)

        return {self.out_label: pred_arr}

    def embed_from_loader(self, loader):
        for item, idx in loader:
            item = item.to(self.device)
            with torch.no_grad():
                audio_emb, _ = torchopenl3.get_audio_embedding(
                    item,
                    self.sr,
                    self.model,
                    center=False,
                    hop_size=self.hop_size,
                )
            # if not same_size:
            #     assert audio_emb.size(0) == 1
            #     audio_emb = audio_emb.squeeze(0)
            yield {self.out_label: audio_emb.cpu().numpy()}, idx
