import webbrowser
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from jinja2 import Environment, PackageLoader, select_autoescape
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from birdclef.models.embedding.tilenet import TileNet
from birdclef.utils import compute_offset, load_audio

ROOT = Path(__file__).parent.parent.parent


@click.group()
def evaluation():
    pass


def intra_cluster(base_path, root, df, model, intra):
    data = []
    for row in df[df.species == intra].sample(50).itertuples():
        try:
            offset = compute_offset(
                row.motif_0,
                row.matrix_profile_window,
                row.duration_cens,
                row.duration_seconds,
            )[0]
        except:
            offset = 0
        y = load_audio(root / row.source_name, offset, 5)
        data.append(y)

    figures = []

    # plot the waveform of the first data point
    for i in range(3):
        z = data[i]
        plt.plot(z)
        name = f"{intra}_intra_waveform_{i}.png"
        figures.append(name)
        plt.savefig(f"{base_path}/{name}")
        plt.clf()
        sf.write(
            base_path / name.replace(".png", ".ogg"),
            z,
            32000,
            format="ogg",
            subtype="vorbis",
        )

    # plot the embedding in an even lower dimension
    emb = model(torch.from_numpy(np.array(data))).detach().numpy()
    g = PCA(n_components=2).fit_transform(emb)

    name = f"{intra}_intraclass.png"
    plt.scatter(g[:, 0], g[:, 1])
    plt.savefig(f"{base_path}/{name}")
    plt.clf()
    figures.append(name)

    kmeans = KMeans(n_clusters=3).fit(emb)
    name = f"{intra}_intraclass_colored.png"
    plt.scatter(g[:, 0], g[:, 1], c=kmeans.labels_)
    plt.savefig(f"{base_path}/{name}")
    plt.clf()
    figures.append(name)

    return figures


def inter_cluster(base_path, root, df, model, species: "list[str]", num_sample=300):
    data = []
    labels = []
    for specie in species:
        for row in df[df.species == specie].sample(num_sample).itertuples():
            try:
                offset = compute_offset(
                    row.motif_0,
                    row.matrix_profile_window,
                    row.duration_cens,
                    row.duration_seconds,
                )[0]
            except:
                offset = 0
            y = load_audio(root / row.source_name, offset, 5)
            data.append(y)
            labels.append(row.species)

    figures = []
    for i in range(3):
        z = data[i]
        plt.plot(z)
        name = f"inter_waveform_{i}.png"
        figures.append(name)
        plt.savefig(f"{base_path}/{name}")
        plt.clf()
        sf.write(
            base_path / name.replace(".png", ".ogg"),
            z,
            32000,
            format="ogg",
            subtype="vorbis",
        )

    emb = model(torch.from_numpy(np.array(data))).detach().numpy()
    le = LabelEncoder().fit(labels)

    g = PCA(n_components=2).fit_transform(emb)
    plt.scatter(g[:, 0], g[:, 1], c=le.transform(labels))
    name = f"interclass.png"
    plt.savefig(f"{base_path}/{name}")
    plt.clf()
    figures.append(name)

    return figures


@evaluation.command()
@click.option(
    "--species",
    default="brnowl,skylar,houfin",
    help="Array of 3 species to perform interclass comparison",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/processed/model/2022-05-17-v7/embedding.ckpt"),
)
@click.option(
    "--parquet",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("data/processed/2022-04-03-motif-consolidated.parquet"),
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False),
    default=Path(f"{ROOT}/data/intermediate/generated_resources"),
)
@click.option(
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default=Path("data/raw/birdclef-2022"),
)
@click.option("--dim", default=512)
@click.option("--experiment-name", default=datetime.utcnow().strftime("%Y%m%d%H%M"))
def clustering(species, checkpoint, parquet, output_path, root, dim, experiment_name):
    output_path = Path(output_path)
    species = species.split(",")

    base_path = output_path / experiment_name
    base_path.mkdir(exist_ok=True, parents=True)

    model = TileNet.load_from_checkpoint(checkpoint, dim=dim)
    df = pd.read_parquet(parquet)
    df["species"] = df.source_name.apply(lambda x: x.split("/")[1])
    df[["species"]].groupby("species").size().sort_values(ascending=False)

    print("Intraclass Figures")
    intra_figures = []
    for specie in species:
        print(f"calculating figures for {specie}")
        intra_figures += intra_cluster(base_path, root, df, model, specie)

    print("Interclass Figures")
    inter_figures = inter_cluster(base_path, root, df, model, species)

    # Create html file with Jinja2 template
    env = Environment(
        loader=PackageLoader("birdclef.workflows", "templates"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("index.html.j2")
    output = template.render(
        species=species,
        intra_figures=intra_figures,
        inter_figures=inter_figures,
    )
    (base_path / "index.html").write_text(output)
    webbrowser.open_new(f"{base_path}/index.html")


if __name__ == "__main__":
    evaluation()
