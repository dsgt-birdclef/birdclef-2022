import numpy as np
import pytest
import soundfile as sf

from birdclef.models.classifier_nn import datasets


@pytest.fixture
def train_root(tmp_path):
    sr = 32000
    for i in range(3):
        sf.write(
            tmp_path / f"{i}.ogg",
            np.ones(3 * 5 * sr) * i,
            sr,
            format="ogg",
            subtype="vorbis",
        )
    return tmp_path


def test_classifier_dataset_interleaves_audio(train_root):
    dataset = datasets.ClassifierDataset(train_root)
    count = 0
    past_two = []
    for item in dataset:
        count += 1
        mean = item.mean()
        assert mean not in past_two
        past_two.append(mean)
        if len(past_two) > 2:
            past_two.pop(0)
    assert count == 3 * 3
