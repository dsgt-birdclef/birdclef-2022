import librosa
import pandas as pd
import pytest
import soundfile as sf
from click.testing import CliRunner

from birdclef.workflows.motif import extract_triplets


@pytest.fixture()
def tile_path(tmp_path):
    return tmp_path


@pytest.fixture()
def metadata_df(tile_path):
    # first chirp example
    # https://librosa.org/doc/main/generated/librosa.chirp.html#librosa.chirp
    sr = 32000
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


# NOTE: we need to be careful that we don't break functionality here, since it
# could break all the tests
@pytest.fixture()
def extract_triplet_path(metadata_df, tile_path, tmp_path):
    # is this the same tmp_path?
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
    return output_path
