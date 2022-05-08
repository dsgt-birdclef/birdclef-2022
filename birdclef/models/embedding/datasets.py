from functools import lru_cache
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision import transforms

from birdclef.utils import slice_seconds


class TileTripletsDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, tile_dir: Path, transform=None):
        self.df = meta_df
        self.tile_dir = Path(tile_dir)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def _load_audio(self, row: pd.Series, col: str, duration=7):
        # build the canonical filename
        offset = int(row[f"{col}_loc"])
        input_path = Path(row[col])
        filename = (
            self.tile_dir / f"{input_path.name.split('.')[0]}_{offset}_{duration}.npy"
        ).as_posix()
        # lets only keep 5 seconds of data
        sr = 32000
        offset = int(np.random.rand() * 2 * sr)
        return np.load(filename)[offset : offset + sr * 5]

    def __getitem__(self, idx: int):
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


@lru_cache(maxsize=32)
def load_audio_cached(path: Path, sr: int = 32000, seconds=5):
    y, _ = librosa.load(path.as_posix(), sr=sr)
    return slice_seconds(y, seconds)


class TileTripletsIterableDataset(IterableDataset):
    """A data loader that generates triplets from the consolidated motif dataset.

    The motif dataset contains the matrix profile and profile index, which
    alongside the tile directory can be used to ensure that we always stream
    audio correctly. There is some trickiness involved to ensure that we can
    split the work across multiple workers.
    """

    def __init__(
        self, motif_consolidated_df: pd.DataFrame, tile_path: Path, transform=None
    ):
        self.df = motif_consolidated_df.sample(frac=1)
        self.tile_path = Path(tile_path)
        self.transform = transform

    def get_motif_pairs(self, start, end, n_queues=32):
        """Find all the motif pairs from all the audio files and put them into
        an iterable. We try to avoid placing pairs next to each other, so that
        we can load mini-batch data in an efficient way.

        n_queues: the number of open audio files to have at a given time
        """
        row_iter = iter(self.df.iloc[start:end].itertuples())

        # set a default value that is true
        open_files = [None]
        while open_files:
            # remove empty elements from the open files
            open_files = [f for f in open_files if f is not None]

            # add new elements to fill up the queue
            while len(open_files) < n_queues and row_iter is not None:
                try:
                    row = next(row_iter)
                except StopIteration:
                    row_iter = None
                    break
                # load the audio
                sliced = load_audio_cached(self.tile_path / row.source_name)
                open_files.append((iter(enumerate(row.pi)), sliced))

            # then yield values from it
            for i in range(open_files):
                pi_iter, sliced = open_files[i]
                try:
                    anchor_idx, neighbor_idx = next(pi_iter)
                    yield dict(anchor=sliced[anchor_idx], neighbor=sliced[neighbor_idx])
                except StopIteration:
                    open_files[i] = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, -1
        else:
            per_worker = int(np.ceil(self.df.shape[0] / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = start + per_worker
        return self.get_motif_pairs(start, end)


class ToFloatTensor:
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


class TileTripletsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_df: pd.DataFrame,
        data_dir: Path,
        batch_size=4,
        num_workers=8,
        shuffle=True,
    ):
        super().__init__()
        self.meta_df = meta_df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None):
        n = self.meta_df.shape[0]
        ratios = [0.9, 0.1]
        lengths = [int(n * p) for p in ratios]
        lengths[0] += n - sum(lengths)

        dataset = TileTripletsDataset(
            self.meta_df,
            self.data_dir,
            transform=transforms.Compose([ToFloatTensor()]),
        )
        self.train, self.val = random_split(dataset, lengths)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=self.shuffle, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.kwargs)

    def test_dataloader(self):
        raise NotImplementedError()
