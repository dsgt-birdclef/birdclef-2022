import random
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from birdclef.models.embedding.tilenet import TileNet
from birdclef.utils import slice_seconds


class ToFloatTensor:
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        z = [torch.from_numpy(z).float() for z in sample]
        if self.device is not None:
            z = [z.to(self.device) for z in z]
        return tuple(z)


class ToEmbedSpace:
    """Converts the samples into embedding space.

    We assume that the batch is on the same device as the model to be trained.
    """

    def __init__(self, checkpoint: Path, z_dim=512):
        self.model = TileNet.load_from_checkpoint(checkpoint, z_dim=z_dim)

    def __call__(self, sample):
        X, y = sample
        device = X.device
        return self.model.to(device)(X), y


class Mixup:
    """Applies mixup to a batch of data

    https://github.com/fastai/fastai_old/blob/488e4fd2939bbe34c0fd0c3867884a9c45877cb9/fastai/callbacks/mixup.py
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        X, y = sample
        lam = (
            torch.from_numpy(np.random.beta(self.alpha, self.alpha, X.size(0)))
            .float()
            .to(X.device)[:, None]
        )
        # lets also generate a random permutation of the possible indices
        perm = torch.randperm(X.size(0))
        X_a = X[perm]
        y_a = y[perm]
        X = X * lam + X_a * (1 - lam)
        y = y * lam + y_a * (1 - lam)
        return X.float(), y.float()


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


class ClassifierSimpleDataset(IterableDataset):
    """A data loader that reads training examples in a linear fashion."""

    def __init__(
        self,
        paths: List[Path],
        label_encoder,
        transform=None,
    ):
        self.paths = paths
        self.label_encoder = label_encoder
        self.transform = transform

    def _slices(self, paths: List[Path], sr=32000):
        """Get all the audio slices for the given audio files."""
        k = len(self.label_encoder.classes_)
        for path in paths:
            # TODO: instead of sliding over full windows, we may want to
            # slide over in increments of 2.5 seconds
            y, _ = librosa.load(path.as_posix(), sr=sr)
            sliced = slice_seconds(y, sr, 5, padding_type="right-align")
            target = self.label_encoder.transform([path.parent.name])
            for _, slice in sliced:
                yield slice, np.identity(k)[target].reshape(-1)

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
        for item in self._slices(self.paths[start:end]):
            if self.transform:
                item = self.transform(item)
            yield item


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root: Path,
        label_encoder,
        embed_checkpoint: Path,
        z_dim: int = 512,
        batch_size=4,
        num_workers=8,
        random_state=None,
        stratify_count=-1,
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

        # change how the dataloader extracts data
        self.stratify_count = stratify_count

        # some on-batch transformations
        self.mixup = Mixup(alpha=0.4)
        self.embed_transform = ToEmbedSpace(embed_checkpoint, z_dim=z_dim)

    def setup(self, stage: Optional[str] = None):
        # NOTE: how important is it to have determinism in the setup?
        random.seed(self.random_state)

        # for training, we choose to use all of the available training data for
        # the specific species
        all_paths = [
            p
            for p in self.train_root.glob("**/*.ogg")
            if p.parent.name in self.label_encoder.classes_
        ]

        # from our notebook, the median track count per species is 15. We'll use
        # this as the target for sampling
        train_paths = []
        for species in self.label_encoder.classes_:
            paths = [p for p in all_paths if p.parent.name == species]
            if not paths:
                continue
            if self.stratify_count > 0:
                # under-sample from over-represented classes and over-sample
                # from under-represented classes.
                paths = random.choices(paths, k=self.stratify_count)
            train_paths += paths

        self.dataset = ClassifierDataset(
            train_paths,
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
            paths = [p for p in all_paths if p.parent.name == species]
            if not paths:
                continue
            val_paths += random.choices(paths, k=1)

        self.val_dataset = ClassifierDataset(
            val_paths,
            self.label_encoder,
            transform=transforms.Compose([ToFloatTensor()]),
            random_state=self.random_state,
        )

    def on_after_batch_transfer(self, batch, idx):
        if self.trainer.training:
            batch = self.mixup(batch)
        return self.embed_transform(batch)

    def train_dataloader(self):
        return DataLoader(self.dataset, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.kwargs)

    def test_dataloader(self):
        raise NotImplementedError()
