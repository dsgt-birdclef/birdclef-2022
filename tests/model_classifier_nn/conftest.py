import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

from birdclef.models.embedding.datasets import TileTripletsIterableDataModule
from birdclef.models.embedding.tilenet import TileNet


@pytest.fixture(scope="session")
def bird_species():
    yield ["foo", "bar", "baz"]


@pytest.fixture(scope="session")
def train_root(tmp_path_factory, bird_species):
    tmp_path = tmp_path_factory.mktemp("train")
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


@pytest.fixture(scope="session")
def label_encoder(bird_species):
    le = LabelEncoder()
    le.fit(["noise"] + bird_species)
    yield le


# now we need to train a checkpoint file
@pytest.fixture(scope="session")
def consolidated_df(train_root):
    yield pd.DataFrame(
        [
            dict(
                source_name=path.as_posix(),
                matrix_profile_window=1,
                pi=[2, 1, 1],
            )
            for path in train_root.glob("**/*.ogg")
        ]
    )


@pytest.fixture(scope="session")
def z_dim():
    yield 64


@pytest.fixture(scope="session")
def model_checkpoint(tmp_path_factory, consolidated_df, train_root, z_dim):
    dm = TileTripletsIterableDataModule(consolidated_df, train_root, num_workers=2)
    model = TileNet(z_dim=z_dim)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)

    tmp_path = tmp_path_factory.mktemp("checkpoint")
    path = tmp_path / "model.chkpt"
    trainer.save_checkpoint(path)
    yield path
