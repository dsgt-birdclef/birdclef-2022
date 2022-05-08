import librosa
import numpy as np
import pytest

from birdclef.utils import load_audio

# TODO: many of these tests are duplicates of test_util


@pytest.mark.parametrize("case", ["3", "10"])
def test_load_audio(metadata_df, tile_path, case):
    row = metadata_df[metadata_df.a == f"{tile_path}/{case}.ogg"].iloc[0]
    sr = 32000
    duration = 7
    y = load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
    assert y.shape == (duration * sr,)


def test_load_audio_short_clip_is_padded(metadata_df, tile_path):
    row = metadata_df[metadata_df.a == f"{tile_path}/3.ogg"].iloc[0]
    sr = 32000
    duration = 7
    y = load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
    assert y.shape == (duration * sr,)
    assert y.sum() > 0
    assert y[:1000].sum() == 0
    assert y[-1:-1000].sum() == 0
    midpoint = y.shape[0] // 2
    assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0


def test_load_audio_short_clip_is_padded(metadata_df, tile_path):
    row = metadata_df[metadata_df.a == f"{tile_path}/3.ogg"].iloc[0]
    sr = 32000
    duration = 7
    y = load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
    assert y.shape == (duration * sr,)
    assert y.sum() > 0
    assert y[:1000].sum() == 0
    assert y[-1000:].sum() == 0
    midpoint = y.shape[0] // 2
    assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0


def test_load_audio_full_clip_is_padded(metadata_df, tile_path):
    row = metadata_df[metadata_df.a == f"{tile_path}/10.ogg"].iloc[0]
    sr = 32000
    duration = 7

    y, _ = librosa.load((tile_path / row.a).as_posix(), sr=sr)
    assert np.abs(y).sum() > 0

    # long audio does not have padding on the edges
    y = load_audio(tile_path / row.a, 0, duration=7, sr=32000)
    assert y.shape == (duration * sr,)
    assert np.abs(y).sum() > 0
    midpoint = y.shape[0] // 2
    assert np.abs(y[:1000]).sum() > 0
    assert np.abs(y[-1000:]).sum() > 0
    assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0

    for x in [-1, -2000]:
        y = load_audio(tile_path / row.a, x, duration=7, sr=32000)
        assert np.abs(y[:1000]).sum() == 0
        assert np.abs(y[-1000:]).sum() > 0
        assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0

    # right padded audio
    for x in [10, 2000]:
        y = load_audio(tile_path / row.a, x, duration=7, sr=32000)
        assert np.abs(y[:1000]).sum() > 0
        assert np.abs(y[-1000:]).sum() == 0
        assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0


def test_extract_triplets(extract_triplet_path):
    # there should be 6 entries
    files = list(extract_triplet_path.glob("*.npy"))
    assert len(files) == 4
