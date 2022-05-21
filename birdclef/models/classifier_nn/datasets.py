from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from birdclef.utils import slice_seconds


class ToFloatTensor:
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class ClassifierDataset(IterableDataset):
    """A data loader that generates training examples from the audio files."""

    def __init__(
        self, training_root: Path, transform=None, random_state: int = 2022, limit=-1
    ):
        self.paths = list(training_root.glob("**/*.ogg"))
        self.transform = transform
        self.limit = limit

        np.random.seed(random_state)
        np.random.shuffle(self.paths)

    def _slices(self, start: int, end: int, n_queues=32, sr=32000):
        """Get all the audio slices for the given audio files.

        n_queues: the number of open audio files to have at a given time
        """
        path_iter = iter(self.paths)

        # set a default value that is true
        open_files = [None]
        while open_files:
            # remove empty elements from the open files
            open_files = [f for f in open_files if f is not None]

            # add new elements to fill up the queue
            while len(open_files) < n_queues and path_iter is not None:
                try:
                    path = next(path_iter)
                except StopIteration:
                    path_iter = None
                    break
                # load the audio
                y, _ = librosa.load(path.as_posix(), sr=sr)
                # TODO: instead of sliding over full windows, we may want to
                # slide over in increments of 2.5 seconds
                sliced = slice_seconds(y, sr, 5, padding_type="right-align")
                if not sliced:
                    continue
                np.random.shuffle(sliced)
                open_files.append(iter(sliced))

            # then yield values from the open files
            for i in range(len(open_files)):
                it = open_files[i]
                try:
                    _, slice = next(it)
                    yield slice
                except StopIteration:
                    open_files[i] = None

    def __iter__(self):
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        # compute number of rows per worker
        rows_per_worker = int(np.ceil(len(self.paths) / num_workers))
        start = worker_id * rows_per_worker
        end = start + rows_per_worker

        count = 0
        for item in self._slices(start, end):
            # only take data from the first worker if we're going to limit data
            if self.limit > 0 and (count >= self.limit or worker_id > 0):
                break
            count += 1
            yield item


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_root: Path,
        batch_size=4,
        num_workers=8,
        random_state=None,
        validation_batches=1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.training_root = training_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = dict(
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.random_state = random_state or np.random.randint(2**31)
        self.validation_batches = validation_batches

    def setup(self, stage: Optional[str] = None):
        self.dataset = ClassifierDataset(
            self.training_root,
            transform=transforms.Compose([ToFloatTensor()]),
            random_state=self.random_state,
            batch_size=self.batch_size,
        )
        self.val_dataset = ClassifierDataset(
            self.training_root,
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
