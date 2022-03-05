import numpy as np
import pytest
from click.testing import CliRunner

from birdclef.workflows.motif import _load_audio, extract_triplets


@pytest.mark.parametrize("case", ["3", "10"])
def test_load_audio(metadata_df, tile_path, case):
    row = metadata_df[metadata_df.a == f"{tile_path}/{case}.ogg"].iloc[0]
    sr = 32000
    duration = 7
    y = _load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
    assert y.shape == (duration * sr,)


def test_load_audio_short_clip_is_padded(metadata_df, tile_path):
    row = metadata_df[metadata_df.a == f"{tile_path}/3.ogg"].iloc[0]
    sr = 32000
    duration = 7
    y = _load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
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
    y = _load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
    assert y.shape == (duration * sr,)
    assert y.sum() > 0
    assert y[:1000].sum() == 0
    assert y[-1:-1000].sum() == 0
    midpoint = y.shape[0] // 2
    assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0


def test_load_audio_full_clip_is_padded(metadata_df, tile_path):
    row = metadata_df[metadata_df.a == f"{tile_path}/10.ogg"].iloc[0]
    sr = 32000
    duration = 7

    # long audio does not have padding on the edges
    y = _load_audio(tile_path / row.a, 5, duration=7, sr=32000)
    assert y.shape == (duration * sr,)
    assert y.sum() > 0
    midpoint = y.shape[0] // 2
    assert np.abs(y[:1000]).sum() > 0
    assert np.abs(y[-1000:]).sum() > 0
    assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0

    # left padded audio
    for x in [-1, 0]:
        y = _load_audio(tile_path / row.a, x, duration=7, sr=32000)
        assert np.abs(y[:1000]).sum() == 0
        assert np.abs(y[-1000:]).sum() > 0
        assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0

    # right padded audio
    for x in [10, 2000]:
        y = _load_audio(tile_path / row.a, x, duration=7, sr=32000)
        assert np.abs(y[:1000]).sum() > 0
        assert np.abs(y[-1000:]).sum() == 0
        assert np.abs(y[midpoint - 500 : midpoint + 500]).sum() > 0


def test_extract_triplets(metadata_df, tile_path, tmp_path):
    df_path = tmp_path / "metadata.parquet"
    metadata_df.to_parquet(df_path)
    output_path = tmp_path / "output"

    runner = CliRunner()
    res = runner.invoke(
        extract_triplets,
        [
            str(x)
            for x in [
                df_path,
                "--dataset-root",
                tile_path,
                "--output",
                output_path,
            ]
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0

    # there should be 6 entries
    files = list(output_path.glob("*.npy"))
    assert len(files) == 6
