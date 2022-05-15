import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from birdclef.models.embedding import datasets


def test_tile_triplets_triplets_dataset(metadata_df, extract_triplet_path):
    dataset = datasets.TileTripletsDataset(metadata_df, extract_triplet_path)
    count = 0
    for i in range(len(dataset)):
        assert dataset[i]
        count += 1
    assert count == metadata_df.shape[0]


def test_tile_triplets_triplets_dataset_batch(metadata_df, extract_triplet_path):
    dataset = datasets.TileTripletsDataset(metadata_df, extract_triplet_path)
    batch_size = 3
    for batch in DataLoader(dataset, batch_size=batch_size, drop_last=True):
        assert "anchor" in batch
        assert "neighbor" in batch
        assert "distant" in batch
        assert isinstance(batch["anchor"], torch.Tensor)
        assert batch["anchor"].shape[0] == batch_size


def test_tile_triplets_datamodule(metadata_df, extract_triplet_path):
    dm = datasets.TileTripletsDataModule(
        metadata_df, extract_triplet_path, batch_size=1
    )
    dm.setup()
    assert len(dm.train_dataloader()) == 9
    assert len(dm.val_dataloader()) == 1
    with pytest.raises(NotImplementedError):
        dm.test_dataloader()


def test_tile_triplets_iterable_dataset_batch_triplet(tile_path, consolidated_df):
    dataset = datasets.TileTripletsIterableDataset(
        consolidated_df, tile_path, batch_size=3
    )
    batch = [
        {
            "anchor": np.ones(3) * 1,
            "neighbor": np.ones(3) * 2,
        },
    ] * 3
    res = dataset._generate_triplets(batch)
    assert set(res.keys()) == set(["anchor", "neighbor", "distant"])
    assert len(res) == 3
    assert res["anchor"].shape[0] == 3
    assert isinstance(res["anchor"], torch.Tensor)


def test_tile_triplets_iterable_dataset_is_batched(tile_path, consolidated_df):
    # TODO: configure the dataset to accept a batch parameter
    batch_size = 3
    dataset = datasets.TileTripletsIterableDataset(
        consolidated_df, tile_path, batch_size=batch_size
    )
    for batch in DataLoader(dataset, num_workers=1, batch_size=None):
        print(batch)
        assert "anchor" in batch
        assert "neighbor" in batch
        assert "distant" in batch
        assert isinstance(batch["anchor"], torch.Tensor)
        assert batch["anchor"].shape == (batch_size, 32000 * 5)


def test_tile_triplets_iterable_dataset_count(tile_path, consolidated_df):
    batch_size = 3
    dataset = datasets.TileTripletsIterableDataset(
        consolidated_df, tile_path, batch_size=batch_size
    )
    count = 0
    for batch in DataLoader(dataset, num_workers=1):
        count += batch["anchor"].shape[0]
    assert count == consolidated_df.shape[0]


def test_tile_triplets_iterable_datamodule(tile_path, consolidated_df):
    batch_size = 3
    dm = datasets.TileTripletsIterableDataModule(
        consolidated_df, tile_path, batch_size=batch_size
    )
    dm.setup()
    # the size of the items in the data loader is equal to the total number of
    # motif pairs in the consolidated df divided by the batch size\
    batch_count = 0
    for _ in dm.train_dataloader():
        batch_count += 1
    assert batch_count == (consolidated_df.shape[0] * 3) / batch_size

    # NOTE: we don't have a validation set
    batch_count = 0
    for _ in dm.val_dataloader():
        batch_count += 1
    assert batch_count == 1

    with pytest.raises(NotImplementedError):
        dm.test_dataloader()
