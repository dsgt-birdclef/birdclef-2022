import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import tqdm
from simple import simple_fast
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_audiomentations import AddColoredNoise, Compose, Gain, PitchShift, Shift

from birdclef.datasets import soundscape_2021
from birdclef.models.embedding.tilenet import TileNet
from birdclef.utils import cens_per_sec


# TODO: only load scored birds
def _load_motif_row(path: Path, sr: int, scored_birds: "list[str]"):
    label = path.parent.name
    y, _ = librosa.load(path, sr=sr)
    if scored_birds:
        label = "other" if label not in scored_birds else label
    return dict(name=path.name.split(".")[0], data=y, label=label)


class MotifDataset(Dataset):
    def __init__(
        self,
        motif_root: Path,
        scored_birds: "list[str]" = [],
        sr: int = 32000,
        limit: int = -1,
        load_other=False,
    ):
        self.sr = sr
        self.scored_birds = scored_birds
        self.paths = list(motif_root.glob("**/*.ogg"))
        if not load_other:
            self.paths = [p for p in self.paths if p.parent.name in self.scored_birds]
        if limit > 0:
            self.paths = random.sample(self.paths, limit)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        motif_row = _load_motif_row(path, self.sr, self.scored_birds)
        # return pd.DataFrame([motif_row])
        return motif_row


def augment_samples(X, batch_size=50, sr=32000):
    """Pass audio samples through augmentation transformation pipeline."""
    apply_augmentation = Compose(
        # TODO: look closer at these specific
        transforms=[
            Gain(),
            PitchShift(
                min_transpose_semitones=-1, max_transpose_semitones=1, sample_rate=sr
            ),
            Shift(min_shift=-0.1, max_shift=0.1),
            AddColoredNoise(),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.utils.data.TensorDataset(torch.unsqueeze(torch.from_numpy(X), 1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    res = []
    for batch in dataloader:
        res.append(
            apply_augmentation(batch[0].to(device), sample_rate=sr)
            .cpu()
            .numpy()
            .squeeze(1)
        )
    return np.concatenate(res)


def _write_audio(name, y, sr=32000):
    sf.write(name, y, sr, format="ogg", subtype="vorbis")


def _write_sampled_data(output, df, num_per_class, sr, parallelism, label):
    samples = df[df.label == label].sample(n=num_per_class, replace=True)
    augmented = augment_samples(np.stack(samples.data.values).astype(np.float32))
    path = output / f"{label}"
    path.mkdir(exist_ok=True, parents=True)

    args = []
    for i, y in enumerate(augmented):
        args.append((path / f"{samples.iloc[i].name}_{i}.ogg", y))
    with Pool(parallelism) as p:
        p.starmap(
            partial(_write_audio, sr=sr), tqdm.tqdm(args, total=len(args)), chunksize=1
        )


def resample_dataset(
    output: Path, df: pd.DataFrame, num_per_class: int = 5000, sr=32000, parallelism=8
):
    """Resample the dataset to have the same number of examples per class."""
    func = partial(_write_sampled_data, output, df, num_per_class, sr, parallelism)
    for label in df.label.unique():
        func(label)


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
    subset = df[df.y == 0].rename(columns={"x": "data", "audio_id": "name"})
    subset["label"] = "other"
    return subset[["name", "data", "label"]]


class NoiseDataset(Dataset):
    def __init__(self, birdclef_2021_root: Path, parallelism=4):
        self.df = load_soundscape_noise(
            birdclef_2021_root=birdclef_2021_root, parallelism=parallelism
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        return dict(
            name=self.df.iloc[idx].name,
            data=self.df.iloc[idx].data,
            label=self.df.iloc[idx].label,
        )


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
