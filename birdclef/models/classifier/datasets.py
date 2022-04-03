import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import tqdm
from simple import simple_fast
from sklearn.model_selection import train_test_split

from birdclef.datasets import soundscape_2021
from birdclef.models.embedding.tilenet import TileNet
from birdclef.utils import cens_per_sec


# TODO: only load scored birds
def _load_motif_row(path: Path, sr: int, scored_birds: "list[str]"):
    label = path.parent.name
    y, _ = librosa.load(path, sr=sr)
    if scored_birds:
        label = "other" if label not in scored_birds else label
    return dict(data=y, label=label)


def load_motif(
    motif_root: Path,
    scored_birds: "list[str]" = [],
    sr: int = 32000,
    parallelism: int = 4,
    limit: int = -1,
    load_other=False,
) -> pd.DataFrame:
    # we can (probably) load this into memory, it's only 600mb of compressed ogg
    # TODO: load iterable chunks of dataframes?
    paths = list(motif_root.glob("**/*.ogg"))
    if not load_other:
        paths = [p for p in paths if p.parent.name in scored_birds]
    if limit > 0:
        paths = random.sample(paths, limit)
    with Pool(parallelism) as p:
        res = p.map(
            partial(_load_motif_row, scored_birds=scored_birds, sr=sr),
            tqdm.tqdm(paths),
            chunksize=1,
        )
    return pd.DataFrame([row for row in res if row])


def _load_ref_motif_row(path: Path, sr: int, cens_sr: int):
    y, _ = librosa.load(path, sr=sr)
    cens = librosa.feature.chroma_cens(y, sr=sr, hop_length=cens_per_sec(sr, cens_sr))
    return dict(cens=cens)


def load_ref_motif(
    motif_root: Path, cens_sr: int = 10, sr: int = 32000, parallellism: int = 4
):
    """Return a dataframe containing CENS features for each reference motif."""
    paths = list(motif_root.glob("**/*.ogg"))
    with Pool(parallellism) as p:
        res = p.map(
            partial(_load_ref_motif_row, sr=sr, cens_sr=cens_sr), paths, chunksize=1
        )
    return pd.DataFrame(res)


def _compute_matrix_profile_features(
    x: np.array,
    df: pd.DataFrame,
    cens_sr: int,
    mp_window: int,
    sr: int = 32000,
) -> np.array:
    cens = librosa.feature.chroma_cens(y=x, sr=sr, hop_length=cens_per_sec(sr, cens_sr))
    res = []
    for row in df.itertuples():
        mp, _ = simple_fast(row.cens, cens, mp_window)
        res += [mp.min(), mp.median(), mp.max()]
    return np.array(res)


def transform_input_motif(
    ref_motif_df: pd.DataFrame,
    X: np.array,
    cens_sr: int = 10,
    mp_window: int = 20,
    sr=32000,
    parallelism=4,
) -> np.array:
    """Return the min, max, and median of the matrix profile for each reference
    motif against each input row.

    We can certainly run out of memory during this computation, so callers
    should take care to batch results.
    """
    func = partial(
        _compute_matrix_profile_features,
        df=ref_motif_df,
        cens_sr=cens_sr,
        mp_window=mp_window,
        sr=sr,
    )
    with Pool(parallelism) as p:
        res = p.map(func, X)
    return np.array(res)


def load_soundscape_noise(birdclef_2021_root: Path, parallelism=4) -> pd.DataFrame:
    """Load noise from soundscape"""
    df = soundscape_2021.load(birdclef_2021_root, parallelism=parallelism)
    subset = df[df.y == 0].rename(columns={"x": "data"})
    subset["label"] = "other"
    return subset[["data", "label"]]


def load_embedding_model(embedding_checkpoint: Path, z_dim: int):
    model = TileNet.load_from_checkpoint(embedding_checkpoint, z_dim=z_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def train_val_test_split(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split(X: np.array, y: np.array, stratify: np.array = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=stratify, train_size=0.9
    )
    return (X_train, y_train), (X_test, y_test)
