import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from birdclef.models.classifier.datasets import (
    MotifDataset,
    augment_samples,
    motif_dataloader,
    resample_dataset,
)


def test_augment_samples_correct_shape():
    sr = 32000
    X = np.random.rand(10, sr).astype(np.float32)
    X_aug = augment_samples(X, batch_size=5, sr=sr)
    assert X_aug.shape == (10, sr)


def test_resample_dataset_correct_shape(tmp_path):
    df = pd.DataFrame(
        [
            {"name": "a", "data": np.random.rand(32000).astype(np.float32), "label": 0},
            {"name": "b", "data": np.random.rand(32000).astype(np.float32), "label": 1},
        ]
    )
    output = tmp_path
    resample_dataset(output, df, 5)
    assert len(list(output.glob("*"))) == 2
    assert len(list(output.glob("*/*.ogg"))) == 10


def test_motif_dataset_correct_length():
    motif_root = Path("data/test")
    scored_birds = json.loads(
        Path("data/raw/birdclef-2022/scored_birds.json").read_text()
    )
    test_motif_dataset = MotifDataset(motif_root=motif_root, scored_birds=scored_birds)
    assert test_motif_dataset.__len__() == 3
