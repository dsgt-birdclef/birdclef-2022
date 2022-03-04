from click.testing import CliRunner

from birdclef.workflows.motif import _load_audio, extract_triplets


def test_load_audio(metadata_df, tile_path):
    row = metadata_df[metadata_df.a == f"{tile_path}/10.ogg"].iloc[0]
    sr = 32000
    duration = 7
    y = _load_audio(tile_path / row.a, row.a_loc, duration=7, sr=32000)
    assert y.shape == (duration * sr,)


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
