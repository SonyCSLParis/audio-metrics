from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

import laion_clap

from audio_metrics.dataset import Embedder, load_audio, GeneratorDataset

# CHECKPOINT = "/home/maarten/audio_metrics/music_audioset_epoch_15_esc_90.14.pt"

# workaround an incompatibility in hugging face transformer def until laion_clap adapts to it.
CHECKPOINT = "/home/maarten/audio_metrics/music_audioset_epoch_15_esc_90.14_no_textpos_ids.pt"

SR = 48000
MODEL = {}

# import laion_clap
# model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
# model.load_ckpt('checkpoint_path/checkpoint_name.pt')


def get_model(checkpoint, device):
    global MODEL
    key = (checkpoint, device)
    if key not in MODEL:
        MODEL[key] = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base"
        ).to(device)
        MODEL[key].load_ckpt(CHECKPOINT)
    return MODEL[key]


class CLAP(Embedder):
    def __init__(self, device):
        super().__init__(sr=SR, mono=True)
        self.model = get_model(CHECKPOINT, device)
        self.device = device
        self.activations = defaultdict(list)
        self.layers = [
            "audio_projection.0",
            "audio_projection.2",
        ]
        self.out_label = "output"
        self.names = self.layers + [self.out_label]
        # for layer in self.layers:
        #     self.model.get_submodule(f"model.{layer}").register_forward_hook(
        #         self._get_activation_hook(layer)
        #     )

    def embed(self, items, same_size=False, batch_size=10):
        audio_sr_pairs = (self.preprocess(item) for item in items)
        self._clear_activations()
        if same_size:
            dataset = GeneratorDataset(
                (audio for audio, _ in audio_sr_pairs),
                multi=False,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size,
                pin_memory=True,
                num_workers=1,
            )
        else:
            dataloader = (
                torch.from_numpy(audio).unsqueeze(0)
                for audio, _ in audio_sr_pairs
            )

        for item in dataloader:
            item = item.to(self.device)
            with torch.no_grad():
                audio_emb = self.model.get_audio_embedding_from_data(
                    x=item, use_tensor=True
                )
            audio_emb = audio_emb.cpu().numpy()
            self.activations[self.out_label].append(audio_emb)

        if same_size:
            for key, acts in self.activations.items():
                self.activations[key] = np.concatenate(acts)[:, None, :]

        result = self.activations
        self._clear_activations()
        return result

    def _register_hooks(self, act_dict):
        hooks = []
        for layer in self.layers:
            hooks.append(
                self.model.get_submodule(
                    f"model.{layer}"
                ).register_forward_hook(
                    self._get_activation_hook(layer, act_dict)
                )
            )
        return hooks

    def embed_from_loader(self, loader):
        acts = {}
        hooks = self._register_hooks(acts)
        for item, idx in loader:
            item = item.to(self.device)
            with torch.no_grad():
                audio_emb = self.model.get_audio_embedding_from_data(
                    x=item, use_tensor=True
                )
            audio_emb = audio_emb.cpu().unsqueeze(1).numpy()
            acts[self.out_label] = audio_emb
            yield acts.copy(), idx
            acts.clear()
        for hook in hooks:
            hook.remove()

    def _clear_activations(self):
        self.activations = defaultdict(list)

    # def _get_activation_hook(self, name):
    #     def hook(model, input, output):
    #         # need clone?
    #         # self.activations[name].append(output.clone().detach().cpu().numpy())
    #         self.activations[name].append(output.detach().cpu().numpy())

    #     return hook

    def _get_activation_hook(self, name, act_dict):
        def hook(model, input, output):
            # need clone?
            # act_dict[name] = output.detach().cpu().unsqueeze(1).numpy()
            act_dict[name] = output.cpu().unsqueeze(1).numpy()

        return hook
