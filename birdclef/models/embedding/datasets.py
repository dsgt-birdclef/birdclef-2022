import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TileTripletsDataset(Dataset):
    def __init__(self, meta_df, tile_dir, transform=None):
        self.df = meta_df
        self.tile_dir = Path(tile_dir)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def _load_audio(self, row, col, duration=5):
        offset = row[f"{col}_loc"]
        filename = self.tile_dir / row[col]
        # -1 is when the audio file is shorter than our 5 second window
        if offset > 0:
            # we know we have enough room to read near the ends, so we can shift
            # it by some amount
            # this is not exactly even, but it's good enough for me right now
            offset = max(offset + (np.random.rand() - 0.5) * duration, 0)
        y, sr = librosa.load(filename, offset=offset, duration=duration)

        # ensure these are audio samples
        length = sr * duration
        return np.resize(np.moveaxis(y, -1, 0), length)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
        except:
            raise KeyError(idx)
        sample = {
            "anchor": self._load_audio(row, "a"),
            "neighbor": self._load_audio(row, "b"),
            "distant": self._load_audio(row, "c"),
        }
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
    meta_df,
    tile_dir: Path,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Returns a DataLoader with ogg data from audio files.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    transform_list = []
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(meta_df, tile_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
