import pytest
import librosa
import soundfile as sf
from birdclef.models.embedding import datasets
import pandas as pd


@pytest.fixture()
def tile_path(tmp_path):
    return tmp_path


@pytest.fixture()
def metadata_df(tile_path):
    # first chirp example
    # https://librosa.org/doc/main/generated/librosa.chirp.html#librosa.chirp
    sr = 22050
    chirp = librosa.chirp(sr=sr, fmin=110, fmax=110 * 64, duration=10)
    for i in [3, 10]:
        sf.write(
            f"{tile_path}/{i}.ogg", chirp[: i * sr], sr, format="ogg", subtype="vorbis"
        )
    return pd.DataFrame(
        [
            {
                "a": f"{tile_path}/{a}.ogg",
                "b": f"{tile_path}/{c}.ogg",
                "c": f"{tile_path}/{b}.ogg",
                "a_loc": a_loc,
                "b_loc": b_loc,
                "c_loc": c_loc,
            }
            for (a, b, c, a_loc, b_loc, c_loc) in [
                # the location must be -1 the track is smaller than the total duration
                [3, 3, 3, -1, -1, -1],
                [10, 10, 10, 0, 5, 9],
            ]
            * 5
        ]
    )


def test_tile_triplets_triplets_dataset(metadata_df, tile_path):
    dataset = datasets.TileTripletsDataset(metadata_df, tile_path)
    count = 0
    for i in range(len(dataset)):
        assert dataset[i]
        count += 1
    assert count == metadata_df.shape[0]


def test_tile_triplets_datamodule(metadata_df, tile_path):
    dm = datasets.TileTripletsDataModule(metadata_df, tile_path, batch_size=1)
    dm.setup()
    assert len(dm.train_dataloader()) == 8
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1
