import numpy as np
import pytest
import soundfile as sf
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def bird_species():
    yield ["foo", "bar", "baz"]


@pytest.fixture
def train_root(tmp_path, bird_species):
    sr = 32000
    for i, bird in enumerate(bird_species):
        path = tmp_path / bird
        path.mkdir()
        for j in range(2):
            sf.write(
                path / f"{j}.ogg",
                np.ones(3 * 5 * sr) * i,
                sr,
                format="ogg",
                subtype="vorbis",
            )
    return tmp_path


@pytest.fixture
def label_encoder(bird_species):
    le = LabelEncoder()
    le.fit(["noise"] + bird_species)
    yield le
