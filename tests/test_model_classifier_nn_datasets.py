import numpy as np
import pytest
import soundfile as sf
import torch
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


def test_classifier_dataset_interleaves_audio(train_root, label_encoder):
    # we have to make sure our label encoder includes "noise" as a label
    k = len(label_encoder.classes_)
    dataset = datasets.ClassifierDataset(
        list(train_root.glob("**/*.ogg")), label_encoder
    )
    count = 0
    sum_labels = np.zeros(k).reshape(-1)
    for item, label in dataset:
        count += 1
        assert label.shape == (k,)
        assert item.shape == (5 * 32000,)
        sum_labels += label
        print(label)
    # we have 3 slices per species, and we're contributing noise at a rate
    # that's proportional to the number of classes in the label_encoder. In this
    # case, the very last element will always be noise since we have to wait for
    # the other generators to finish. We note that all the generators for the
    # audio files are the same length, so they finish at the same time.
    assert count == (k * 3 + 1) * 2, sum_labels
    assert (
        np.abs((sum_labels - (np.ones(k) * 3 * 2 + np.array([0, 0, 0, 2])))).sum() == 0
    )


def test_classifier_datamodule(train_root, label_encoder):
    batch_size = 4
    # for this test, we drop the any remaining elements in the dataset, which in
    # the above example ends up being 2 extra elements outside of the batch size
    dm = datasets.ClassifierDataModule(
        train_root, label_encoder, batch_size=batch_size, num_workers=1, drop_last=True
    )
    dm.setup()
    batch_count = 0
    for X, y in dm.train_dataloader():
        assert isinstance(X, torch.Tensor)
        assert X.shape == (batch_size, 5 * 32000)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (batch_size, len(label_encoder.classes_))
        batch_count += 1
    assert batch_count == 6

    batch_count = 0
    for _ in dm.val_dataloader():
        batch_count += 1
    assert batch_count == 3

    with pytest.raises(NotImplementedError):
        dm.test_dataloader()
