import warnings

import click
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from birdclef.models.embedding import datasets, tilenet


@click.group()
def embed():
    pass


@embed.command()
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/raw/birdclef-2022",
)
@click.option("--dim", type=int, default=64)
@click.option("--n-mels", type=int, default=64)
@click.option("--name", type=str, default="tile2vec")
@click.option("--log-dir", type=str, default="data/intermediate/tb_logs")
@click.option("--checkpoint-dir", type=str, default="data/intermediate/checkpoint")
def fit(metadata, dataset_dir, dim, n_mels, name, log_dir, checkpoint_dir):
    metadata_df = pd.read_parquet(metadata)
    data_module = datasets.TileTripletsDataModule(
        metadata_df,
        dataset_dir,
        batch_size=4,
        num_workers=8,
    )
    model = tilenet.TileNet(z_dim=dim, n_mels=n_mels)

    trainer = pl.Trainer(
        gpus=-1,
        precision=16,
        # This ends up with a size of 4
        auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        logger=TensorBoardLogger(log_dir, name=name),
        default_root_dir=checkpoint_dir,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        profiler="simple",
    )
    trainer.tune(model, data_module)
    print(f"batch size: {data_module.batch_size}, lr: {model.lr}")
    # trainer.fit(model, data_module)


if __name__ == "__main__":
    embed()
