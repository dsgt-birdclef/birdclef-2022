import datetime
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelEncoder
from torchsummary import summary

from birdclef.models.classifier_nn.callbacks import InputMonitor
from birdclef.models.classifier_nn.datasets import ClassifierDataModule
from birdclef.models.classifier_nn.model import ClassifierNet


@click.group()
def classify_nn():
    pass


@classify_nn.command()
@click.option("--output", type=click.Path(exists=False, file_okay=False))
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False),
    default=Path("data/intermediate/classify-nn"),
)
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw/birdclef-2022/train_audio"),
)
@click.option(
    "--embedding-checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=Path(
        "data/intermediate/embedding/tile2vec-v5/version_10/checkpoints/"
        "epoch=2-step=5635.ckpt"
    ),
)
@click.option("--dim", type=int, default=512)
@click.option(
    "--filter-set",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/raw/birdclef-2022/scored_birds.json"),
)
@click.option("--parallelism", type=int, default=8)
def fit(
    output,
    root_dir,
    dataset_dir,
    embedding_checkpoint,
    dim,
    filter_set,
    parallelism,
):
    ver = version("birdclef")
    if output is None:
        ds = datetime.datetime.utcnow().strftime("%Y%m%d%H%m")
        output = Path(f"data/processed/classify-nn/{ver}-{ds}")
    output = Path(output)
    root_dir = Path(root_dir)
    root_dir.mkdir(exist_ok=True, parents=True)

    filtered_birds = json.loads(Path(filter_set).read_text())
    label_encoder = LabelEncoder()
    label_encoder.fit(["noise"] + filtered_birds)
    dm = ClassifierDataModule(
        Path(dataset_dir),
        label_encoder,
        Path(embedding_checkpoint),
        dim,
        batch_size=32,
        num_workers=parallelism,
    )
    model = ClassifierNet(dim, len(label_encoder.classes_))
    trainer = pl.Trainer(
        gpus=-1,
        auto_lr_find=True,
        default_root_dir=root_dir,
        logger=TensorBoardLogger(root_dir, name=ver, log_graph=True),
        detect_anomaly=True,
        max_epochs=100,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                auto_insert_metric_name=True,
                save_top_k=5,
                train_time_interval=datetime.timedelta(minutes=15),
            ),
            InputMonitor(),
        ],
    )
    trainer.tune(model, dm)
    print(f"batch size: {dm.batch_size}, lr: {model.lr}")
    summary(model, torch.randn(1, dim).cpu())
    trainer.fit(model, dm)

    output.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output / "classify.ckpt")
    shutil.copy(embedding_checkpoint, output / "embedding.ckpt")
    (output / "metadata.json").write_text(
        json.dumps(
            dict(
                embedding_source=Path(embedding_checkpoint).as_posix(),
                embedding_dim=dim,
                created=datetime.datetime.now().isoformat(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    classify_nn()
