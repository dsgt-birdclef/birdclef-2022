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
@click.option("--dim", type=int, default=64)
@click.option("--n-mels", type=int, default=64)
def model_summary(metadata, dataset_dir, dim, n_mels):
    metadata_df = pd.read_parquet(metadata)
    data_module = datasets.TileTripletsDataModule(
        metadata_df,
        dataset_dir,
        batch_size=25,
        num_workers=4,
    )
    model = tilenet.TileNet(z_dim=dim, n_mels=n_mels)
    trainer = pl.Trainer(
        gpus=-1,
        precision=16,
        fast_dev_run=True,
        # callbacks=[CheckBatchGradient()],
    )
    trainer.fit(model, data_module)
    summary(model, model.example_input_array)


@embed.command()
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--dim", type=int, default=64)
@click.option("--n-mels", type=int, default=64)
@click.option("--name", type=str, default="tile2vec")
@click.option("--log-dir", type=str, default="data/intermediate/tb_logs")
@click.option("--checkpoint-dir", type=str, default="data/intermediate/checkpoint")
@click.option("--limit-train-batches", type=int, default=5000)
@click.option("--limit-val-batches", type=int, default=500)
def fit(
    metadata,
    dataset_dir,
    dim,
    n_mels,
    name,
    log_dir,
    checkpoint_dir,
    limit_train_batches,
    limit_val_batches,
):
    metadata_df = pd.read_parquet(metadata)
    data_module = datasets.TileTripletsDataModule(
        metadata_df,
        dataset_dir,
        # batch size of 64 is roughly as large as we can go with 16 bit
        # precision, 32 for for 32 bit precision
        batch_size=20,
        # setting this number higher can lead to slower results, oddly enough
        num_workers=4,
    )
    model = tilenet.TileNet(z_dim=dim, n_mels=n_mels)

    trainer = pl.Trainer(
        gpus=-1,
        # using 16-bit precision causes issues with finding the learning rate
        # precision=16,
        # auto_scale_batch_size="binsearch",
        auto_lr_find=True,
        # NOTE: does thi set the checkpoint directory correctly?
        default_root_dir=checkpoint_dir,
        logger=TensorBoardLogger(log_dir, name=name),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        detect_anomaly=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            # NOTE: need to figure out how to change the model so that it
            # actually passes this batch gradient condition.
            # CheckBatchGradient(),
            ModelCheckpoint(filename="tilenet-{epoch:02d}-{val_loss:.2f}"),
        ],
        # profiler="simple",
    )
    trainer.tune(model, data_module)
    print(f"batch size: {data_module.batch_size}, lr: {model.lr}")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    embed()
