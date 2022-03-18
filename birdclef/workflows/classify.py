import datetime
import json
import pickle
import shutil
from importlib.resources import read_text
from pathlib import Path

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from birdclef.models import classifier
from birdclef.utils import transform_input


@click.group()
def classify():
    pass


class SubmitClassifier:
    def __init__(self, label_encoder, onehot_encoder, classifier):
        self.label_encoder = label_encoder
        self.onehot_encoder = onehot_encoder
        self.classifier = classifier


@classify.command()
@click.argument("output", type=click.Path(exists=False, file_okay=False))
@click.option(
    "--birdclef-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw/birdclef-2021"),
)
@click.option(
    "--motif-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/intermediate/2022-03-12-extracted-primary-motif"),
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
@click.option(
    "--filter-set",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/raw/birdclef-2022/scored_birds.json"),
)
@click.option("--limit", type=int, default=-1)
def train(
    birdclef_root, output, motif_root, embedding_checkpoint, dim, filter_set, limit
):
    scored_birds = json.loads(Path(filter_set).read_text())
    df = pd.concat(
        [
            # TODO: loading in batches
            classifier.load_motif(
                Path(motif_root), scored_birds=scored_birds, limit=limit
            ),
            classifier.load_soundscape_noise(Path(birdclef_root)),
        ]
    )

    le = LabelEncoder()
    le.fit(df.label)
    ohe = OneHotEncoder()
    ohe.fit(le.transform(df.label).reshape(-1, 1))

    model, device = classifier.load_embedding_model(embedding_checkpoint, dim)
    X_raw = np.stack(df.data.values)
    X = transform_input(model, device, X_raw)
    y = le.transform(df.label.values)

    train_pair, val_pair, (X_test, y_test) = classifier.train_val_test_split(X, y)

    bst = classifier.train(
        lgb.Dataset(*train_pair),
        lgb.Dataset(*val_pair),
        le.classes_.shape[0],
    )
    print(f"best number of iterations: {bst.best_iteration}")

    # TODO: better scoring
    score = f1_score(
        ohe.transform(y_test.reshape(-1, 1)),
        ohe.transform(np.argmax(bst.predict(X_test), axis=1).reshape(-1, 1)),
        average="macro",
    )
    print(f"test score: {score}")

    # write things to disk in a directory to make this as easy as possible to
    # load
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = SubmitClassifier(le, ohe, bst)
    with open(output / "submit_classifier.pkl", "wb") as fp:
        pickle.dump(model, fp)

    # copy over the checkpoint file for easy loading
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


@classify.command()
@click.argument("output")
@click.option(
    "--classifier-source", required=True, type=click.Path(exists=True, file_okay=False)
)
def predict(output, classifier_source):
    classifier_source = Path(classifier_source)
    with open(classifier_source / "submit_classifier.pkl", "rb") as fp:
        model = pickle.load(fp)

    metadata = json.loads((classifier_source / "metadata.json").read_text())
    model, device = classifier.load_embedding_model(
        classifier_source / "embedding.ckpt", metadata["embedding_dim"]
    )


if __name__ == "__main__":
    classify()
