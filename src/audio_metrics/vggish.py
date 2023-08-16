from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from audio_metrics.dataset import Embedder, load_audio

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
        super().__init__()
        self.model, self.sr = get_vggish_model(device)

    def preprocess(self, item):
        audio, sr = super().preprocess(item)
        # compute spectrogram
        data = self.model._preprocess(audio, sr)
        return data

    def embed(self, dataset):
        return {"last_feature_layer": get_activations(dataset, self.model)}
        


def get_activations(dataset, model, batch_size=50, num_workers=1):
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
    if num_workers is None:
        num_workers = 0
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, pin_memory=True, num_workers=num_workers
    )
    pred_arr = []
    device = next(model.parameters()).device
    with torch.no_grad():
        # for batch in tqdm(dataloader):
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.cpu().numpy()
            pred_arr.append(pred)
    pred_arr = np.concatenate(pred_arr)
    return pred_arr
