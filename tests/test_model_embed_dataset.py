import librosa
import pandas as pd
import pytest
import soundfile as sf

from birdclef.models.embedding import datasets


def test_tile_triplets_triplets_dataset(metadata_df, extract_triplet_path):
    dataset = datasets.TileTripletsDataset(metadata_df, extract_triplet_path)
    count = 0
    for i in range(len(dataset)):
        assert dataset[i]
        count += 1
    assert count == metadata_df.shape[0]


def test_tile_triplets_datamodule(metadata_df, extract_triplet_path):
    dm = datasets.TileTripletsDataModule(
        metadata_df, extract_triplet_path, batch_size=1
    )
    dm.setup()
    assert len(dm.train_dataloader()) == 9
    assert len(dm.val_dataloader()) == 1
    with pytest.raises(NotImplementedError):
        dm.test_dataloader()


@pytest.fixture()
def consolidated_df(tile_path):
    sr = 32000
    n = 10
    chirp = librosa.chirp(sr=sr, fmin=110, fmax=110 * 64, duration=15)
    for i in range(n):
        sf.write(f"{tile_path}/{i}.ogg", chirp, sr, format="ogg", subtype="vorbis")
    return pd.DataFrame(
        [{"source_name": f"{i}.ogg", "pi": [2, 1, 1]} for i in range(n)]
    )


def test_tile_triplets_iterable_dataset_is_batched(tile_path, consolidated_df):
    # TODO: configure the dataset to accept a batch parameter
    batch_size = 2
    dataset = datasets.TileTripletsIterableDataset(consolidated_df, tile_path)
    count = 0
    for batch in dataset:
        # TODO: assert the size of the batch tensor is correct
        for item in batch:
            assert "anchor" in item
            assert "neighbor" in item
            assert "distant" in item
            count += 1
    assert count == consolidated_df.shape[0] * 3


def test_tile_triplets_iterable_datamodule(tile_path, consolidated_df):
    batch_size = 1
    dm = datasets.TileTripletsIterableDataModule(
        consolidated_df, tile_path, batch_size=batch_size
    )
    dm.setup()
    # the size of the items in the data loader is equal to the total number of
    # motif pairs in the consolidated df divided by the batch size
    assert len(dm.train_dataloader()) == (consolidated_df.shape[0] * 3) / batch_size

    # NOTE: we only keep a few batches (a fixed number) for validating results.
    # It should be a small fraction of the total dataset size.
    assert len(dm.val_dataloader()) == 1
    with pytest.raises(NotImplementedError):
        dm.test_dataloader()
