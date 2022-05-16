from datetime import timedelta
from pathlib import Path

import click
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from birdclef.models.embedding import datasets, tilenet


# https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch
# TODO: add this as a proper test
class CheckBatchGradient(pl.Callback):
    def on_train_start(self, trainer, model):
        n = 0
        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()

        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)

        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")


@click.group()
def embed():
    pass


@embed.command(name="summary")
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset-dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--datamodule", type=click.Choice(["iterable", "legacy"]), default="iterable"
)
@click.option("--dim", type=int, default=512)
@click.option("--n-mels", type=int, default=64)
def model_summary(metadata, dataset_dir, datamodule, dim, n_mels):
    metadata_df = pd.read_parquet(metadata)
    module = (
        datasets.TileTripletsIterableDataModule
        if datamodule == "iterable"
        else datasets.TileTripletsDataModule
    )
    data_module = module(
        metadata_df,
        dataset_dir,
        batch_size=20,
        num_workers=4,
        validation_batches=50,
    )
    model = tilenet.TileNet(z_dim=dim, n_mels=n_mels)
    trainer = pl.Trainer(
        gpus=-1,
        # precision=16,
        fast_dev_run=True,
        # callbacks=[CheckBatchGradient()],
    )
    trainer.fit(model, data_module)
    summary(model, model.example_input_array)


@embed.command()
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset-dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--datamodule", type=click.Choice(["iterable", "legacy"]), default="iterable"
)
@click.option("--dim", type=int, default=512)
@click.option("--n-mels", type=int, default=64)
@click.option("--name", type=str, default="tile2vec-v4")
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False),
    default=Path("data/intermediate/embedding"),
)
@click.option("--limit-train-batches", type=int, default=None)
@click.option("--limit-val-batches", type=int, default=None)
@click.option("--max-epochs", type=int, default=20)
@click.option("--checkpoint", type=str)
def fit(
    metadata,
    dataset_dir,
    datamodule,
    dim,
    n_mels,
    name,
    root_dir,
    limit_train_batches,
    limit_val_batches,
    max_epochs,
    checkpoint,
):
    root_dir = Path(root_dir)
    metadata_df = pd.read_parquet(metadata)
    module = (
        datasets.TileTripletsIterableDataModule
        if datamodule == "iterable"
        else datasets.TileTripletsDataModule
    )
    data_module = module(
        metadata_df,
        dataset_dir,
        # With the 900k param model at 16 bits, apparently this can go up to
        # 449959. I don't trust this value though, and empirically 100 per batch
        # fills up gpu memory quite nicely.
        # The default model has 20m parameters which will take much longer to
        # finish.
        batch_size=64,
        num_workers=8,
        validation_batches=50,
    )
    if checkpoint:
        model = tilenet.TileNet.load_from_checkpoint(
            root_dir / name / checkpoint, z_dim=dim, n_mels=n_mels
        )
    else:
        model = tilenet.TileNet(z_dim=dim, n_mels=n_mels)

    trainer = pl.Trainer(
        gpus=-1,
        # using 16-bit precision causes issues with finding the learning rate,
        # and there are often anomalies: RuntimeError: Function 'SqrtBackward0'
        # returned nan values in its 0th output.
        # precision=16,
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        default_root_dir=root_dir / "root",
        logger=TensorBoardLogger(root_dir, name=name, log_graph=True),
        limit_train_batches=limit_train_batches or 1.0,
        limit_val_batches=limit_val_batches or 1.0,
        detect_anomaly=True,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            # NOTE: need to figure out how to change the model so that it
            # actually passes this batch gradient condition.
            # CheckBatchGradient(),
            ModelCheckpoint(
                monitor="val_loss",
                auto_insert_metric_name=True,
                save_top_k=5,
                train_time_interval=timedelta(minutes=15),
            ),
        ],
        # profiler="simple",
    )
    # trainer.tune(model, data_module)
    print(f"batch size: {data_module.batch_size}, lr: {model.lr}")
    summary(model, model.example_input_array)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    embed()
