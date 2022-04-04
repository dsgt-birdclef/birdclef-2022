import datetime
import json
import pickle
import shutil
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from birdclef.datasets import soundscape
from birdclef.models.classifier import datasets
from birdclef.models.classifier import model as classifier_model
from birdclef.utils import chunks, transform_input


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
    default=Path("data/intermediate/2022-04-03-extracted-top-motif"),
)
@click.option(
    "--filter-set",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/raw/birdclef-2022/scored_birds.json"),
)
@click.option("--limit", type=int, default=-1)
@click.option("--num-per-class", type=int, default=2500)
@click.option("--parallelism", type=int, default=8)
def prepare_dataset(
    output, birdclef_root, motif_root, filter_set, limit, num_per_class, parallelism
):
    scored_birds = json.loads(Path(filter_set).read_text())

    df = pd.concat(
        [
            datasets.load_motif(
                Path(motif_root),
                scored_birds=scored_birds,
                limit=limit,
                parallelism=parallelism,
            ),
            datasets.load_soundscape_noise(
                Path(birdclef_root), parallelism=parallelism
            ),
        ]
    )
    datasets.resample_dataset(Path(output), df, num_per_class)


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
    "--ref-motif-root",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/intermediate/2022-03-18-motif-sample-k-64-v1"),
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
@click.option("--cens-sr", type=int, default=10)
@click.option("--mp-window", type=int, default=20)
@click.option("--limit", type=int, default=-1)
@click.option("--parallelism", type=int, default=8)
def train(
    birdclef_root,
    output,
    motif_root,
    ref_motif_root,
    embedding_checkpoint,
    dim,
    filter_set,
    cens_sr,
    mp_window,
    limit,
    parallelism,
):
    scored_birds = json.loads(Path(filter_set).read_text())
    # load the reference motif dataset
    ref_motif_df = datasets.load_ref_motif(Path(ref_motif_root), cens_sr=cens_sr)

    df = pd.concat(
        [
            datasets.load_motif(
                Path(motif_root),
                scored_birds=scored_birds,
                limit=limit,
                parallelism=parallelism,
            ),
            datasets.load_soundscape_noise(
                Path(birdclef_root), parallelism=parallelism
            ),
        ]
    )
    if limit > 0:
        df = df.iloc[:limit]

    le = LabelEncoder()
    le.fit(df.label)
    ohe = OneHotEncoder()
    ohe.fit(le.transform(df.label).reshape(-1, 1))

    model, device = datasets.load_embedding_model(embedding_checkpoint, dim)
    X_raw = np.stack(df.data.values)

    # transform data in batches, mostly because transforming the motif features
    # takes up a significant amount of memory. Windows will complain about not
    # being able to fit the memory into a page segment.
    batch_size = 50
    X = None
    for chunk in tqdm.tqdm(chunks(X_raw, batch_size), total=len(X_raw) // batch_size):
        transformed_chunk = np.hstack(
            [
                transform_input(model, device, chunk, batch_size=batch_size),
                datasets.transform_input_motif(
                    ref_motif_df,
                    chunk,
                    cens_sr=cens_sr,
                    mp_window=mp_window,
                    parallelism=parallelism,
                ),
            ]
        )
        if X is None:
            X = transformed_chunk
        else:
            # stack the X with the transformed chunk
            X = np.vstack([X, transformed_chunk])
    print(f"done transforming data: {X.shape}")

    y = (
        ohe.transform(le.transform(df.label.values).reshape(-1, 1))
        .toarray()
        .astype(int)
    )

    train_pair, (X_test, y_test) = datasets.split(
        X,
        y,
        # not enough examples
        # stratify=le.transform(df.label)
    )

    print("training classifier")
    bst = classifier_model.train(train_pair)

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
    model = classifier_model.SubmitClassifier(le, ohe, bst)
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
                cens_sr=cens_sr,
                mp_window=mp_window,
            ),
            indent=2,
        )
    )
    motif_output = output / "reference_motifs"
    if motif_output.exists():
        shutil.rmtree(motif_output)
    shutil.copytree(ref_motif_root, motif_output)


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
    model, device = datasets.load_embedding_model(
        classifier_source / "embedding.ckpt", metadata["embedding_dim"]
    )
    ref_motif_df = datasets.load_ref_motif(
        classifier_source / "reference_motifs", cens_sr=metadata["cens_sr"]
    )

    test_df = pd.read_csv(Path(birdclef_root) / "test.csv")
    print(test_df.head())

    res = []
    for df in soundscape.load_test_soundscapes(
        Path(birdclef_root) / "test_soundscapes"
    ):
        X_raw = np.stack(df.x.values)
        X = np.hstack(
            [
                transform_input(model, device, X_raw),
                datasets.transform_input_motif(
                    ref_motif_df,
                    X_raw,
                    cens_sr=metadata["cens_sr"],
                    mp_window=metadata["mp_window"],
                ),
            ]
        )
        y_pred = cls_model.classifier.predict(X)
        res_inner = []
        for row, pred in zip(df.itertuples(), y_pred):
            labels = cls_model.label_encoder.inverse_transform(np.flatnonzero(pred))
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
    classify()
