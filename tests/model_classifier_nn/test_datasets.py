import numpy as np
import pytest
import torch

from birdclef.models.classifier_nn import datasets


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


@pytest.mark.parametrize("num_workers", [1, 4, 8])
def test_classifier_datamodule_multiple_workers(train_root, label_encoder, num_workers):
    dm = datasets.ClassifierDataModule(
        train_root, label_encoder, batch_size=1, num_workers=num_workers, drop_last=True
    )
    dm.setup()
    batch_count = 0
    for X, y in dm.train_dataloader():
        assert isinstance(X, torch.Tensor)
        assert X.shape == (1, 5 * 32000)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, len(label_encoder.classes_))
        batch_count += 1

    # figure out the range for the number of extra samples that are being added to the data
    k = len(label_encoder.classes_)
    assert batch_count >= (2 * (k)) * 3 + num_workers
    assert batch_count < (2 * (k + 1)) * 3 + num_workers * 2
