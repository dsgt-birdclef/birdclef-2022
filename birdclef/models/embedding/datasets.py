import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TileTripletsDataset(Dataset):
    def __init__(self, tile_dir, transform=None, n_triplets=None, pairs_only=True):
        self.tile_dir = Path(tile_dir)

        self.tile_files = list(
            set([".".join(p.name.split(".")[:2]) for p in self.tile_dir.glob("*.ogg")])
        )
        self.transform = transform
        self.n_triplets = n_triplets
        self.pairs_only = pairs_only

    def __len__(self):
        if self.n_triplets:
            return self.n_triplets
        else:
            return len(self.tile_files)

    def __getitem__(self, idx):
        name = self.tile_files[idx]
        a, _ = librosa.load(self.tile_dir / f"{name}.0.ogg")
        n, _ = librosa.load(self.tile_dir / f"{name}.1.ogg")
        if self.pairs_only:
            distant = np.random.choice(self.tile_files)
            d_idx = np.random.randint(0, 2)
            d, _ = librosa.load(self.tile_dir / f"{distant}.{d_idx}.ogg")
        else:
            raise NotImplementedError("triples do not contain a distant pair")

        # TODO: do not assume audio is this length...
        # ensure these are all the same
        sample_rate = 22050
        seconds = 5
        length = sample_rate * seconds

        a = np.resize(np.moveaxis(a, -1, 0), length)
        n = np.resize(np.moveaxis(n, -1, 0), length)
        d = np.resize(np.moveaxis(d, -1, 0), length)

        sample = {"anchor": a, "neighbor": n, "distant": d}
        if self.transform:
            sample = self.transform(sample)
        return sample


### TRANSFORMS ###


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __call__(self, sample):
        a, n, d = (
            torch.from_numpy(sample["anchor"]).float(),
            torch.from_numpy(sample["neighbor"]).float(),
            torch.from_numpy(sample["distant"]).float(),
        )
        sample = {"anchor": a, "neighbor": n, "distant": d}
        return sample


### TRANSFORMS ###


def triplet_dataloader(
    tile_dir,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    n_triplets=None,
    pairs_only=True,
):
    """
    Returns a DataLoader with ogg data from audio files.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    transform_list = []
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(
        tile_dir, transform=transform, n_triplets=n_triplets, pairs_only=pairs_only
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
