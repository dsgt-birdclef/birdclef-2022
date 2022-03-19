import datetime
import json
import pickle
import shutil
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from birdclef.datasets import soundscape
from birdclef.models import classifier
from birdclef.utils import transform_input


@click.group()
def classify():
    pass


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
    y = (
        ohe.transform(le.transform(df.label.values).reshape(-1, 1))
        .toarray()
        .astype(int)
    )

    train_pair, (X_test, y_test) = classifier.split(X, y)

    bst = classifier.train(train_pair)

    score = f1_score(
        y_test,
        bst.predict(X_test),
        average="macro",
    )
    print(f"test score: {score}")

    # write things to disk in a directory to make this as easy as possible to
    # load
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = classifier.SubmitClassifier(le, ohe, bst)
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
    "--birdclef-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw/birdclef-2022"),
)
@click.option(
    "--classifier-source", required=True, type=click.Path(exists=True, file_okay=False)
)
def predict(output, birdclef_root, classifier_source):
    classifier_source = Path(classifier_source)
    with open(classifier_source / "submit_classifier.pkl", "rb") as fp:
        cls_model = pickle.load(fp)

    metadata = json.loads((classifier_source / "metadata.json").read_text())
    print(metadata)
    model, device = classifier.load_embedding_model(
        classifier_source / "embedding.ckpt", metadata["embedding_dim"]
    )

    test_df = pd.read_csv(Path(birdclef_root) / "test.csv")
    print(test_df.head())

    res = []
    for df in soundscape.load_test_soundscapes(
        Path(birdclef_root) / "test_soundscapes"
    ):
        X_raw = np.stack(df.x.values)
        X = transform_input(model, device, X_raw)
        y_pred = cls_model.classifier.predict(X)
        res_inner = []
        for row, pred in zip(df.itertuples(), y_pred):
            labels = cls_model.label_encoder.inverse_transform(
                np.nonzero(pred.reshape(-1))
            )
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
    submission_df[["row_id", "target"]].to_csv(output, index=False)


if __name__ == "__main__":
    classify()
