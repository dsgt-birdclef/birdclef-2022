import random
from pathlib import Path
from typing import List, Optional

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
        return tuple([torch.from_numpy(z).float() for z in sample])


class ClassifierDataset(IterableDataset):
    """A data loader that generates training examples from the audio files."""

    def __init__(
        self,
        paths: List[Path],
        label_encoder,
        transform=None,
        random_state: int = 2022,
    ):
        self.paths = paths
        self.label_encoder = label_encoder
        self.transform = transform

        random.seed(random_state)
        np.random.seed(random_state)
        np.random.shuffle(self.paths)

    def _slices(self, paths: List[Path], n_queues=32, sr=32000):
        """Get all the audio slices for the given audio files.

        This dataloader will also generate random noise in every batch, how
        frequently it occurs is related to the number of queues that are open at
        any given time.

        n_queues: the number of open audio files to have at a given time
        """
        path_iter = iter(paths)
        k = len(self.label_encoder.classes_)

        def noise_generator():
            # a generator that continuously yields noise
            while True:
                yield np.random.randn(sr * 5)

        open_files = [(noise_generator(), "noise")]
        while True:
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
                open_files.append(
                    (
                        (x for _, x in sliced),
                        path.parent.name,
                    )
                )

            # only the noise generator is left
            if len(open_files) <= 1:
                return

            # then yield values from the open files
            for i in range(len(open_files)):
                it, label = open_files[i]
                try:
                    slice = next(it)
                    # https://stackoverflow.com/a/49790223
                    target = self.label_encoder.transform([label])
                    yield slice, np.identity(k)[target].reshape(-1)
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

        k = len(self.label_encoder.classes_)
        for item in self._slices(self.paths[start:end], n_queues=k):
            if self.transform:
                item = self.transform(item)
            yield item


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root: Path,
        label_encoder,
        batch_size=4,
        num_workers=8,
        random_state=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.train_root = train_root
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )
        self.random_state = random_state or np.random.randint(2**31)

    def setup(self, stage: Optional[str] = None):
        # NOTE: how important is it to have determinism in the setup?
        random.seed(self.random_state)

        # for training, we choose to use all of the available training data for
        # the specific species
        self.dataset = ClassifierDataset(
            list(self.train_root.glob("**/*.ogg")),
            self.label_encoder,
            transform=transforms.Compose([ToFloatTensor()]),
            random_state=self.random_state,
        )

        # for the validation dataset, we only use a subset of the files. We
        # assume the length of a single file in the training dataset is roughly
        # the same length (although in practice there are clips that are over 10
        # minutes in length)
        val_paths = []
        for species in self.label_encoder.classes_:
            paths = list(self.train_root.glob(f"{species}/*.ogg"))
            if not paths:
                continue
            val_paths.append(random.choice(paths))

        self.val_dataset = ClassifierDataset(
            val_paths,
            self.label_encoder,
            transform=transforms.Compose([ToFloatTensor()]),
            random_state=self.random_state,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.kwargs)

    def test_dataloader(self):
        raise NotImplementedError()