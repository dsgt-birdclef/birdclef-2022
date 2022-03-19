from multiprocessing import Pool
from pathlib import Path

import librosa
import pandas as pd
import tqdm

from birdclef.utils import chunks, slice_seconds


def parse_soundscape(path: Path, sr=32000, window=5) -> pd.DataFrame:
    """Convert a soundscape into the expected labeling format."""
    y, _ = librosa.load(path, sr=sr)
    df = pd.DataFrame(slice_seconds(y, sr, window), columns=["end_time", "x"])
    df["file_id"] = path.name.split(".ogg")[0]
    return df[["file_id", "end_time", "x"]]


def load_test_soundscapes(
    test_root: Path, chunk_size=4, parallelism=4
) -> "Generator[pd.DataFrame]":
    """Yield data ready for prediction.

    Audio are 1 minute clips broken into 5 second windows with a 32khz sample
    rate.
    """
    paths = list(test_root.glob("*"))
    chunked_paths = list(chunks(paths, chunk_size))
    with Pool(parallelism) as p:
        for path_chunk in tqdm.tqdm(chunked_paths):
            yield pd.concat(p.map(parse_soundscape, path_chunk))
