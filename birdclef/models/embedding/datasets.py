from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.stats import mode
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
from torchvision import transforms

from birdclef.utils import chunks, slice_seconds


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


class TileTripletsIterableDataset(IterableDataset):
    """A data loader that generates triplets from the consolidated motif dataset.

    The motif dataset contains the matrix profile and profile index, which
    alongside the tile directory can be used to ensure that we always stream
    audio correctly. There is some trickiness involved to ensure that we can
    split the work across multiple workers.
    """

    def __init__(
        self,
        motif_consolidated_df: pd.DataFrame,
        tile_path: Path,
        batch_size=32,
        transform=None,
        random_state: int = 2022,
        limit=-1,
    ):
        self.min_batch_size = 3
        self.df = motif_consolidated_df.sample(frac=1, random_state=random_state)
        self.tile_path = Path(tile_path)
        self.transform = transform or transforms.Compose([ToFloatTensor()])
        self.batch_size = batch_size
        if self.batch_size < self.min_batch_size:
            raise ValueError(f"Batch size must be at least {self.min_batch_size}")
        self.limit = limit

    def _cens_to_seconds_mode(self, pi, cens_window):
        """Convert the profile index to seconds and mode.

        pi: the profile index
        cens_window: the window size in seconds
        """
        return [mode(chunk)[0][0] for chunk in chunks(pi // cens_window, cens_window)]

    def get_motif_pairs(self, start: int, end: int, n_queues=32):
        """Find all the motif pairs from all the audio files and put them into
        an iterable. We try to avoid placing pairs next to each other, so that
        we can load mini-batch data in an efficient way.

        n_queues: the number of open audio files to have at a given time
        """
        error_count = 0
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
                sr = 32000
                y, _ = librosa.load(
                    (self.tile_path / row.source_name).as_posix(), sr=sr
                )
                sliced = slice_seconds(y, sr, 5)
                if not sliced:
                    continue
                # shuffle the indices
                indices = list(
                    enumerate(
                        self._cens_to_seconds_mode(
                            np.array(row.pi), row.matrix_profile_window
                        )
                    )
                )
                np.random.shuffle(indices)
                open_files.append([iter(indices), sliced])

            # then yield values from it
            for i in range(len(open_files)):
                pi_iter, sliced = open_files[i]
                try:
                    anchor_idx, neighbor_idx = next(pi_iter)
                    row = dict(
                        anchor=sliced[anchor_idx][1], neighbor=sliced[neighbor_idx][1]
                    )
                    yield row
                except StopIteration:
                    open_files[i] = None
                except IndexError:
                    # This is because we drop the last index during
                    # slice_seconds, so we get into situations where the motif
                    # pair doesn't exist
                    error_count += 1
                    continue
        if error_count > 0:
            print(f"encountered {error_count} errors")

    def _generate_triplets(self, batch):
        """Generate triplets from a batch of motif pairs."""
        batch_len = len(batch)

        # TODO: this shuffling method is (potentially) slow because it
        # requires iterating over all the batches and retrying to make
        # sure that we don't have any accidentally matching pairs.
        res = []
        for i, row in enumerate(batch):
            selector = ["anchor", "neighbor"][np.random.randint(2)]
            # choose an index that's not the current batch index
            while True:
                j = np.random.randint(0, batch_len)
                if i != j:
                    break
            d = self.transform(dict(**row, distant=batch[j][selector]))
            res.append(d)

        # and now we have to create an object that has all of the tensors
        # batched up
        return dict(
            anchor=torch.stack([row["anchor"] for row in res]),
            neighbor=torch.stack([row["neighbor"] for row in res]),
            distant=torch.stack([row["distant"] for row in res]),
        )

    def _batch_triplet(self, iter, batch_size):
        """Generate a new iterable that contains batch_size elements"""
        batch = []
        for row in iter:
            batch.append(row)
            if len(batch) == batch_size:
                yield self._generate_triplets(batch)
                batch = []
        # note that we drop anything that isn't the full batch size, so we don't
        # have to deal with the degenerate case where this is only a single item
        # in the batch. This is equivalent to `drop_last` in the data loader

    def __iter__(self):
        # Tried handling this via worker_init_fn() but I couldn't get slicing to
        # work since the dataset copied into each worker is a full
        # TileTripletsIterableDataset entity...
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()

        # compute number of rows per worker
        rows_per_worker = int(
            np.ceil(self.df.shape[0] / float(worker_info.num_workers))
        )

        # compute start and end of dataset for each worker
        worker_id = worker_info.id
        start = worker_id * rows_per_worker
        end = start + rows_per_worker

        count = 0
        for item in self._batch_triplet(
            self.get_motif_pairs(start, end), self.batch_size
        ):
            # only take data from the first worker if we're going to limit data
            if self.limit > 0 and (count >= self.limit or worker_id > 0):
                break
            count += 1
            yield item


class TileTripletsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_df: pd.DataFrame,
        data_dir: Path,
        batch_size=4,
        num_workers=8,
        shuffle=True,
        *args,
        **kwargs,
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


# TODO: implement the majority of the functionality here
class TileTripletsIterableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        motif_consolidated_df: pd.DataFrame,
        data_dir: Path,
        batch_size=4,
        num_workers=8,
        random_state=None,
        validation_batches=1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.df = motif_consolidated_df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = dict(
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.random_state = random_state or np.random.randint(2**31)
        self.validation_batches = validation_batches

    def setup(self, stage: Optional[str] = None):
        # generate a random number to seed the dataset
        self.dataset = TileTripletsIterableDataset(
            self.df,
            self.data_dir,
            transform=transforms.Compose([ToFloatTensor()]),
            random_state=self.random_state,
            batch_size=self.batch_size,
        )
        # TODO: Do we still need to do train/test splits here?
        self.val_dataset = TileTripletsIterableDataset(
            self.df,
            self.data_dir,
            transform=transforms.Compose([ToFloatTensor()]),
            random_state=self.random_state,
            batch_size=self.batch_size,
            limit=self.validation_batches,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.kwargs)

    def test_dataloader(self):
        raise NotImplementedError()
