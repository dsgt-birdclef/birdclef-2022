import numpy as np
import pytest
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

from birdclef.models.classifier_nn import datasets


@pytest.fixture
def bird_species():
    yield ["foo", "bar", "baz"]


@pytest.fixture
def train_root(tmp_path, bird_species):
    sr = 32000
    for i, bird in enumerate(bird_species):
        path = tmp_path / bird
        path.mkdir()
        sf.write(
            path / f"{i}.ogg",
            np.ones(3 * 5 * sr) * i,
            sr,
            format="ogg",
            subtype="vorbis",
        )
    return tmp_path


def test_classifier_dataset_interleaves_audio(train_root, bird_species):
    # we have to make sure our label encoder includes "noise" as a label
    le = LabelEncoder()
    le.fit(bird_species + ["noise"])
    k = len(le.classes_)
    dataset = datasets.ClassifierDataset(train_root, le)
    count = 0
    past_two = []
    sum_labels = np.zeros(k).reshape(-1)
    for item, label in dataset:
        count += 1
        assert label.shape == (k,)
        assert item.shape == (5 * 32000,)
        sum_labels += label
        print(label)
        mean = item.mean()
        assert mean not in past_two
        past_two.append(mean)
        if len(past_two) > 2:
            past_two.pop(0)
    # we have 3 slices per species, and we're contributing noise at a rate
    # that's proportional to the number of classes in the label_encoder. In this
    # case, the very last element will always be noise since we have to wait for
    # the other generators to finish.
    assert count == k * 3 + 1
    assert np.abs((sum_labels - (np.ones(k) * 3 + np.array([0, 0, 0, 1])))).sum() == 0
