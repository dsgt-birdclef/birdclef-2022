from multiprocessing import Pool
from pathlib import Path

import librosa
import pandas as pd
import tqdm

from birdclef.utils import slice_seconds


def parse_metadata(path: Path) -> dict:
    audio_id, site, _ = path.name.split("_")
    return dict(audio_id=audio_id, site=site)


def parse_soundscape(path: Path, sr=32000, window=5) -> pd.DataFrame:
    """Convert a soundscape into the expected labeling format."""
    y, _ = librosa.load(path, sr=sr)
    df = pd.DataFrame(slice_seconds(y, sr, window), columns=["seconds", "x"])
    for key, value in parse_metadata(path).items():
        df[key] = value
    df["row_id"] = df.apply(
        lambda row: f"{row.audio_id}_{row.site}_{row.seconds}", axis=1
    )
    return df[["row_id", "audio_id", "site", "seconds", "x"]]


def load_training_soundscapes(train_root: Path, parallelism: int = 8) -> pd.DataFrame:
    res = []
    paths = list(train_root.glob("*"))
    with Pool(parallelism) as p:
        res = p.map(parse_soundscape, tqdm.tqdm(paths), chunksize=1)
    return pd.concat(res)


def load(birdclef_root: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(birdclef_root / "train_soundscape_labels.csv")
    data_df = load_training_soundscapes(birdclef_root / "train_soundscapes")

    # merge the two datasets together to get the raw signal and label
    df = labels_df.merge(data_df[["row_id", "x"]], on="row_id")
    df["y"] = (df.birds != "nocall").astype(int)
    return df[["x", "y"]]
