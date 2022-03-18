import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split

from birdclef.datasets import soundscape
from birdclef.models.embedding.tilenet import TileNet


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
    parallelism: int = 8,
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


def load_soundscape_noise(birdclef_2021_root: Path) -> pd.DataFrame:
    """Load noise from soundscape"""
    df = soundscape.load(birdclef_2021_root)
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


def train(train_ds: tuple, val_ds: tuple, num_class: int, **kwargs):
    # TODO: parameter sweep
    param = {
        "num_leaves": 31,
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "softmax",
        **kwargs,
    }
    num_boost_round = 100
    # TODO: train with multioutputclassifier
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier
    bst = lgb.train(
        param,
        train_ds,
        num_boost_round,
        valid_sets=val_ds,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )
    return bst
