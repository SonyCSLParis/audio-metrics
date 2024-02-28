from pathlib import Path
from tqdm import tqdm

import numpy as np
import einops
import torch

from audio_metrics.dataset import Embedder, load_audio  # , GeneratorDataset

# TODO: store model for reuse


def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def get_vggish_model(device=None, reset_weights=False):
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    if device is not None:
        device = torch.device(
            device
        )  # dirty hack since the model moves itself to gpu if it exists anyways, oh well
        model.to(device)
        model.device = device
        torch.cuda.empty_cache()
    model.eval()
    model.postprocess = False
    model.preprocess = False
    model.embeddings[5] = torch.nn.Identity()
    if reset_weights:
        print("resetting weights")
        model.apply(weight_reset)
    sr = 16000
    return model, sr


class VGGish(Embedder):
    def __init__(self, device):
        self.device = device
        self.model, sr = get_vggish_model(device)
        super().__init__(sr=sr, mono=True)
        self.names = ["last_feature_layer"]

    def embed(self, items, same_size=False, batch_size=10):
        # items can be an iterable of audio filepaths, or of (audio, sr) pairs
        # in both cases self.preprocess(item) will ensure the audio is at the
        # correct samplerate
        audio_sr_pairs = (self.preprocess(item) for item in items)
        mel_specs = (self.model._preprocess(*item) for item in audio_sr_pairs)
        return {
            "last_feature_layer": get_activations(
                mel_specs, self.model, batch_size, same_size
            )
        }

    def embed_from_loader(self, loader):
        with torch.no_grad():
            for audio_batch, idx in loader:
                batch_size = len(audio_batch)
                spec_batch = [
                    self.model._preprocess(audio, self.sr)
                    for audio in audio_batch.cpu().numpy()
                ]
                spec_batch = einops.rearrange(
                    spec_batch, "b t 1 w h -> (b t) 1 w h"
                )
                spec_batch = spec_batch.to(self.device)
                audio_emb = self.model(spec_batch)
                audio_emb = einops.rearrange(
                    audio_emb, "(b t) d -> b t d", b=batch_size
                )
                yield {self.names[0]: audio_emb.cpu().numpy()}, idx
            # if not same_size:
            #     assert audio_emb.size(0) == 1
            #     audio_emb = audio_emb.squeeze(0)


class GeneratorDataset(torch.utils.data.IterableDataset):
    """Turn a generator that yields model preprocess output into an IterableDataset"""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.lens = []

    def __iter__(self):
        for items in self.generator:
            self.lens.append(len(items))
            for item in items:
                yield item


def get_activations(items, model, batch_size, same_size, num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataset     : Dataset
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    # if num_workers is None:
    # num_workers = 0
    dataset = GeneratorDataset(items)
    # WARNING: num_workers != 0 should not be used on an iterabledataset (will
    # lead to duplication of items)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, pin_memory=True, num_workers=0
    )
    pred_arr = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.cpu().numpy()
            pred_arr.append(pred)
    pred_arr = np.concatenate(pred_arr)
    pred_arr = np.split(pred_arr, np.cumsum(dataset.lens[:-1]))
    if same_size:
        pred_arr = np.stack(pred_arr, 0)
    return pred_arr
