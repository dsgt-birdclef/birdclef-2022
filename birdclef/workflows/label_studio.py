import json
from multiprocessing import Pool
from pathlib import Path

import click
import librosa
import lightgbm as lgb
import numpy as np
import torch
import tqdm

from birdclef.models.embedding.tilenet import TileNet
from birdclef.utils import chunks

from .nocall import transform_input


@click.group()
def label_studio():
    pass


@label_studio.command()
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--prefix", type=str, default="http://localhost:8000")
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw"),
)
@click.option("--pattern", type=str, default="birdclef-2022/train_audio/**/*.ogg")
def train_list(output, prefix, input, pattern):
    files = list(Path(input).glob(pattern))
    Path(output).write_text(
        "\n".join([f"{prefix}/{file.relative_to(input).as_posix()}" for file in files])
    )


def _load_audio(file):
    y, _ = librosa.load(file, sr=32000)
    return y


@label_studio.command()
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--prefix", type=str, default="http://localhost:8000")
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data"),
)
@click.option(
    "--pattern",
    type=str,
    default="intermediate/2022-03-12-extracted-primary-motif/**/*.ogg",
)
@click.option(
    "--nocall-params",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/intermediate/2022-03-12-lgb.txt"),
)
@click.option(
    "--embedding-checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=Path(
        "data/intermediate/embedding/tile2vec-v2/version_1/checkpoints/"
        "epoch=2-step=10872.ckpt"
    ),
)
@click.option(
    "--filter-set",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/raw/birdclef-2022/scored_birds.json"),
)
@click.option("--dim", type=int, default=64)
def motif_list(
    output, prefix, input, pattern, nocall_params, embedding_checkpoint, dim, filter_set
):
    filter_set = set(json.loads(Path(filter_set).read_text()))
    files = [
        file for file in Path(input).glob(pattern) if file.parent.name in filter_set
    ]

    # this is absolutely the wrong place for this training loop to exist, but oh
    # well...
    model = TileNet.load_from_checkpoint(embedding_checkpoint, z_dim=dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    bst = lgb.Booster(model_file=nocall_params)

    chunk_size = 100
    # keep this number under the physical number of cores to avoid issues
    # outlined here: https://stackoverflow.com/a/66814746
    parallelism = 6
    res = []
    for chunk in tqdm.tqdm(list(chunks(files, chunk_size))):
        with Pool(parallelism) as p:
            audio_data = p.map(_load_audio, chunk)
        emb = transform_input(model, device, np.stack(audio_data))
        pred = bst.predict(emb)
        for file, score in zip(chunk, pred):
            # https://labelstud.io/guide/tasks.html#Example-JSON-format
            res.append(
                dict(
                    data=dict(
                        audio=f"{prefix}/{file.relative_to(input).as_posix()}",
                        audio_id=file.name.split(".")[0],
                        species=file.parent.name,
                        score=score,
                    )
                )
            )
    Path(output).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    label_studio()
