from pathlib import Path

import click
import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from birdclef.datasets import soundscape_2021
from birdclef.models.embedding.tilenet import TileNet
from birdclef.utils import transform_input


@click.group()
def nocall():
    pass


@nocall.command()
@click.argument("output")
@click.option(
    "--birdclef-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw/birdclef-2021"),
)
@click.option(
    "--embedding-checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=Path(
        "data/intermediate/embedding/tile2vec-v2/version_1/checkpoints/"
        "epoch=2-step=10872.ckpt"
    ),
)
@click.option("--dim", type=int, default=64)
def fit_soundscape_cv(output, birdclef_root, embedding_checkpoint, dim):
    df = soundscape_2021.load(Path(birdclef_root))
    model = TileNet.load_from_checkpoint(embedding_checkpoint, z_dim=dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    emb = transform_input(model, device, np.stack(df.x.values))

    X_train, X_test, y_train, y_test = train_test_split(
        emb, df.y.values, train_size=0.9
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    param = {"num_leaves": 31, "objective": "binary"}
    param["metric"] = "auc"

    num_boost_round = 100
    bst = lgb.cv(
        param,
        train_data,
        num_boost_round,
        nfold=5,
        callbacks=[lgb.early_stopping(stopping_rounds=5)],
        return_cvbooster=True,
    )

    print("best number of iterations: " + str(bst["cvbooster"].best_iteration))

    bst["cvbooster"].save_model(
        output,
        num_iteration=bst["cvbooster"].best_iteration,
    )

    for i, pred in enumerate(bst["cvbooster"].predict(X_test)):
        print(f"test score for cv {i}: {roc_auc_score(y_test, pred)}")


@nocall.command()
@click.argument("output")
@click.option(
    "--birdclef-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw/birdclef-2021"),
)
@click.option(
    "--embedding-checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=Path(
        "data/intermediate/embedding/tile2vec-v2/version_1/checkpoints/"
        "epoch=2-step=10872.ckpt"
    ),
)
@click.option("--dim", type=int, default=64)
def fit_soundscape(output, birdclef_root, embedding_checkpoint, dim):
    df = soundscape_2021.load(Path(birdclef_root))
    model = TileNet.load_from_checkpoint(embedding_checkpoint, z_dim=dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    emb = transform_input(model, device, np.stack(df.x.values))

    X_train, X_test, y_train, y_test = train_test_split(
        emb, df.y.values, train_size=0.85
    )
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    param = {"num_leaves": 31, "objective": "binary"}
    param["metric"] = "auc"

    num_boost_round = 100
    bst = lgb.train(
        param,
        train_data,
        num_boost_round,
        valid_sets=val_data,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )

    print("best number of iterations: " + str(bst.best_iteration))

    bst.save_model(
        output,
        num_iteration=bst.best_iteration,
    )

    print(f"test score: {roc_auc_score(y_test, bst.predict(X_test))}")


if __name__ == "__main__":
    nocall()
