"""Find the motif pair for each training audio clip in a dataset.

This is the first step of the preprocessing pipeline.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import click
import librosa
import numpy as np
import tqdm
from simple import simple_fast

from birdclef.utils import cens_per_sec

ROOT = Path(__file__).parent.parent.parent


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
    data, sample_rate = librosa.load(input_path)
    duration = librosa.get_duration(y=data, sr=sample_rate)
    if duration < 5:
        # TODO: write a proper error
        print(f"{input_path} - duration is too small: {duration}")
        return

    cens = librosa.feature.chroma_cens(
        y=data, sr=sample_rate, hop_length=cens_per_sec(sample_rate, cens_sr)
    )
    mp, pi = simple_fast(cens, cens, mp_window)
    motif = np.argmin(mp)
    idx = int(motif), int(pi[motif])

    path.mkdir(exist_ok=True, parents=True)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "source_name": "/".join(input_path.parts[-3:]),
                "cens_sample_rate": cens_sr,
                "matrix_profile_window": mp_window,
                "motif_0": idx[0],
                "motif_1": idx[1],
                "sample_rate": sample_rate,
                "duration_cens": cens.shape[1],
                "duration_samples": data.shape[0],
                "duration_seconds": round(duration, 2),
            },
            indent=2,
        )
        + "\n"
    )
    np.save(f"{path}/mp.npy", mp)
    np.save(f"{path}/pi.npy", pi)


@click.command()
@click.option("--species", type=str)
def main(species):
    rel_root = Path(ROOT / "data/raw/birdclef-2022/train_audio")
    src = rel_root
    if species:
        src = src / species
    dst = Path(ROOT / "data/intermediate/2022-02-21-motif")

    files = list(src.glob("**/*.ogg"))
    if not files:
        raise ValueError("no files found")

    args = []
    for path in files:
        rel_dir = path.relative_to(rel_root).parent
        args.append([path, dst / rel_dir, 10, 50])

    with Pool(12) as p:
        p.starmap(write, tqdm.tqdm(args, total=len(args)), chunksize=1)


if __name__ == "__main__":
    main()
