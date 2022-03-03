"""Find the motif pair for each training audio clip in a dataset.

This is the first step of the preprocessing pipeline.
"""
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import click
import librosa
import numpy as np
import pandas as pd
import tqdm
from simple import simple_fast

from birdclef.utils import cens_per_sec

ROOT = Path(__file__).parent.parent.parent

# UserWarning: n_fft=1024 is too small for input signal of length=846
warnings.filterwarnings("ignore", ".*n_fft.*")


def write(input_path, output_path, cens_sr=10, mp_window=50):
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_path should be a folder")

    # new directory for each set of files
    name = input_path.name.split(".ogg")[0]
    path = Path(output_path) / name

    # exit early if all the data already exists
    if path.exists() and path.is_dir():
        if all([(path / x).exists() for x in ["metadata.json", "mp.npy", "pi.npy"]]):
            return

    # read the audio file and calculate the position of the motif
    data, sample_rate = librosa.load(input_path, 32000)
    duration = librosa.get_duration(y=data, sr=sample_rate)
    cens = librosa.feature.chroma_cens(
        y=data, sr=sample_rate, hop_length=cens_per_sec(sample_rate, cens_sr)
    )

    metadata = {
        "source_name": "/".join(input_path.parts[-3:]),
        "cens_sample_rate": cens_sr,
        "matrix_profile_window": mp_window,
        "sample_rate": sample_rate,
        "duration_cens": cens.shape[1],
        "duration_samples": data.shape[0],
        "duration_seconds": round(duration, 2),
        "motif_0": None,
        "motif_1": None,
    }
    path.mkdir(exist_ok=True, parents=True)
    if duration < 5:
        # the duration is too short, but let's still write out useful data
        # print(f"{input_path} - duration is too small: {duration}")
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return

    mp, pi = simple_fast(cens, cens, mp_window)
    motif = np.argmin(mp)
    idx = int(motif), int(pi[motif])

    (path / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                **{"motif_0": idx[0], "motif_1": idx[1]},
            },
            indent=2,
        )
        + "\n"
    )
    np.save(f"{path}/mp.npy", mp)
    np.save(f"{path}/pi.npy", pi)


@click.group()
def motif():
    pass


@motif.command()
@click.option("--species", type=str)
@click.option("--output", type=str, default="2022-02-21-motif")
def extract(species, dataset):
    rel_root = Path(ROOT / "data/raw/birdclef-2022/train_audio")
    src = rel_root
    if species:
        src = src / species
    dst = Path(ROOT / f"data/intermediate/{output}")

    files = list(src.glob("**/*.ogg"))
    if not files:
        raise ValueError("no files found")

    args = []
    for path in files:
        rel_dir = path.relative_to(rel_root).parent
        args.append([path, dst / rel_dir, 10, 50])

    with Pool(12) as p:
        p.starmap(write, tqdm.tqdm(args, total=len(args)), chunksize=1)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@motif.command()
@click.option("--input", type=str, default="2022-02-21-motif")
@click.option("--output", type=str, default="2022-02-26-motif-consolidated")
def consolidate(input, output):
    src = Path(ROOT / f"data/intermediate/{input}")
    dst = Path(ROOT / f"data/intermediate/{output}.parquet")

    files = list(src.glob("**/metadata.json"))
    if not files:
        raise ValueError("no files found")

    with Pool(12) as p:
        data = p.map(_read_json, tqdm.tqdm(files, total=len(files)), chunksize=1)

    df = pd.DataFrame(data)
    print(df.head())
    df.to_parquet(dst)


def generate_samples(df, n_samples, grouping_col="family", window_sec=5):
    """
    We generate two sets of dataset, and then union them at the end first
    generate the same of completely random samples. This will comprise of half
    our distant data. The other half will comprise of random samples from
    motifs. We will always choose to use stratified sampling in order to
    represent all species equally.

    there's no reason that our embedding should try to cluster between classes
    based on the frequency of samples. We should instead try to embed based on
    the actual content of the samples that we hear. This is why this function
    will perform stratified sampling over the family that the audio comes from.
    """
    res = pd.DataFrame()
    groups = df[grouping_col].unique()
    for group in groups:

        def sample_group(k, include=True):
            return (
                df[df[grouping_col] == group]
                if include
                else df[df[grouping_col] != group]
            ).sample(k, replace=True)

        k = n_samples // len(groups) // 2

        # inter_clip
        x, y, z = [sample_group(k, True).fillna(-1).reset_index() for _ in range(3)]

        tmp_ab = pd.concat(
            [
                pd.DataFrame(
                    {
                        "a": x.source_name,
                        "a_loc": x.motif_0,
                        "b": x.source_name,
                        "b_loc": x.motif_1,
                    }
                ),
                pd.DataFrame(
                    {
                        "a": y.source_name,
                        "a_loc": y.motif_0,
                        "b": z.source_name,
                        "b_loc": z.motif_1,
                    }
                ),
            ]
        )
        # now we randomly sample against clips outside the family which are
        # (motifs, random)

        x, y = [sample_group(k, False).fillna(-1).reset_index() for _ in range(2)]
        tmp_c = pd.concat(
            [
                pd.DataFrame({"c": x.source_name, "c_loc": x.motif_0}),
                pd.DataFrame(
                    {
                        "c": y.source_name,
                        "c_loc": y.duration_seconds.apply(
                            lambda s: -1
                            if s <= window_sec
                            else np.random.rand() * (s - window_sec) + (window_sec / 2)
                        ),
                    }
                ),
            ]
        )

        tmp = pd.concat([tmp_ab, tmp_c], axis=1)

        if res.empty:
            res = tmp
        else:
            res = pd.concat([res, tmp])
    return res.sample(frac=1).reset_index()


@motif.command()
@click.option("--input", type=str, default="2022-02-26-motif-consolidated")
@click.option("--output", type=str, default="2022-02-26-motif-triplets")
@click.option("--samples", type=float, default=1e6)
def generate_triplets(input, output, samples):
    samples = int(samples)
    taxa_path = Path(ROOT / "data/raw/birdclef-2022/eBird_Taxonomy_v2021.csv")
    src = Path(ROOT / f"data/intermediate/{input}.parquet")
    dst = Path(ROOT / f"data/intermediate/{output}-{samples:.0e}.parquet")

    taxa = pd.read_csv(taxa_path)
    df = pd.read_parquet(src)
    df["species"] = df.source_name.apply(lambda x: x.split("/")[1]).astype(str)
    transformed_df = df.merge(
        taxa[["SPECIES_CODE", "FAMILY"]].rename(
            columns={"SPECIES_CODE": "species", "FAMILY": "family"}
        ),
        on="species",
        how="left",
    )[["source_name", "species", "family", "motif_0", "motif_1", "duration_seconds"]]

    res = generate_samples(transformed_df, samples)
    print(res)
    res.to_parquet(dst)


if __name__ == "__main__":
    motif()
