import datetime
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelEncoder
from torchsummary import summary

from birdclef.datasets import soundscape
from birdclef.models.classifier_nn.callbacks import InputMonitor
from birdclef.models.classifier_nn.datasets import ClassifierDataModule, ToEmbedSpace
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
@click.option(
    "--stratify-count",
    type=int,
    default=15,
    help="Stratify the number of tracks used during training for each species.",
)
@click.option(
    "--queue-size",
    type=int,
    default=8,
    help="Limit the number of concurrent audio files open.",
)
@click.option(
    "--step-size",
    type=int,
    default=2,
    help="Limit the number of concurrent audio files open.",
)
@click.option("--parallelism", type=int, default=8)
def fit(
    output,
    root_dir,
    dataset_dir,
    embedding_checkpoint,
    dim,
    filter_set,
    stratify_count,
    queue_size,
    step_size,
    parallelism,
):
    ver = version("birdclef")
    if output is None:
        ds = datetime.datetime.utcnow().strftime("%Y%m%d%H%M")
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
        stratify_count=stratify_count,
        queue_size=queue_size,
        step_size=step_size,
        batch_size=32,
        num_workers=parallelism,
    )
    model = ClassifierNet(dim, len(label_encoder.classes_))
    trainer = pl.Trainer(
        gpus=-1,
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
    summary(model, torch.randn(1, dim).cpu())
    trainer.fit(model, dm)

    print(f"saving to {output}")
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output / "classify.ckpt")
    shutil.copy(embedding_checkpoint, output / "embedding.ckpt")
    (output / "metadata.json").write_text(
        json.dumps(
            dict(
                embedding_source=Path(embedding_checkpoint).as_posix(),
                embedding_dim=dim,
                created=datetime.datetime.now().isoformat(),
                stratify_count=stratify_count,
                queue_size=queue_size,
                step_size=step_size,
            ),
            indent=2,
        )
    )


@classify_nn.command()
@click.argument("output")
@click.option(
    "--birdclef-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw/birdclef-2022"),
)
@click.option(
    "--classifier-source", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option("--method", type=click.Choice(["top", "top-not-noise"]), default="top")
def predict(output, birdclef_root, classifier_source, method):
    birdclef_root = Path(birdclef_root)
    classifier_source = Path(classifier_source)

    filter_set = json.loads((birdclef_root / "scored_birds.json").read_text())
    label_encoder = LabelEncoder()
    label_encoder.fit(["noise"] + filter_set)

    metadata = json.loads((classifier_source / "metadata.json").read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_embed = ToEmbedSpace(
        classifier_source / "embedding.ckpt", z_dim=metadata["embedding_dim"]
    )
    model = ClassifierNet.load_from_checkpoint(
        classifier_source / "classify.ckpt",
        z_dim=metadata["embedding_dim"],
        n_classes=len(label_encoder.classes_),
    )

    test_df = pd.read_csv(Path(birdclef_root) / "test.csv")
    print(test_df.head())

    res = []
    for df in soundscape.load_test_soundscapes(
        Path(birdclef_root) / "test_soundscapes"
    ):
        X_raw = torch.from_numpy(np.stack(df.x.values)).float().to(device)
        X, _ = to_embed((X_raw, None))
        y_pred = model.to(device)(X).cpu().detach().numpy()
        # now convert the prediction to something that we can use
        res_inner = []
        for row, pred in zip(df.itertuples(), y_pred):
            labels = []
            sorted_indices = np.argsort(pred)[::-1]
            if method == "top":
                labels = label_encoder.inverse_transform(sorted_indices[:1])
            elif method == "top-not-noise":
                for label in label_encoder.inverse_transform(sorted_indices[1:]):
                    if label == "noise":
                        break
                    labels.append(label)
            for label in labels:
                res_inner.append(
                    {
                        "file_id": row.file_id,
                        "bird": label,
                        "end_time": row.end_time,
                        "target": True,
                    }
                )
        res.append(pd.DataFrame(res_inner))
    res_df = pd.concat(res)
    submission_df = test_df.merge(
        res_df[res_df.bird != "other"], on=["file_id", "bird", "end_time"], how="left"
    ).fillna(False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(submission_df.head())
    submission_df[["row_id", "target"]].to_csv(output, index=False)


if __name__ == "__main__":
    classify_nn()
