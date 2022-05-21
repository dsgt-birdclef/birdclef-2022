import numpy as np
import pytest
import pytorch_lightning as pl
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from torchsummary import summary

from birdclef.models.classifier_nn.datasets import ClassifierDataModule
from birdclef.models.classifier_nn.model import ClassifierNet

# TODO: move this module in a shared location
from birdclef.workflows.embed import CheckBatchGradient


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


@pytest.mark.parametrize("z_dim", [64, 113])
def test_tilenet_train(train_root, label_encoder, z_dim):
    dm = ClassifierDataModule(train_root, label_encoder, batch_size=5, num_workers=1)
    model = ClassifierNet(z_dim=z_dim, n_classes=len(label_encoder.classes_))
    trainer = pl.Trainer(fast_dev_run=True, callbacks=[CheckBatchGradient()])
    trainer.fit(model, dm)

    metrics = trainer.callback_metrics
    print(metrics)
    assert np.abs(metrics["loss"].detach()) > 0

    # assert the shape of the data
    for batch in dm.val_dataloader():
        x = batch["anchor"]
    summary(model, x)
