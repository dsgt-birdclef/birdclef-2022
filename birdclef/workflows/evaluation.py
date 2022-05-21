import os
import webbrowser
from datetime import date
from pathlib import Path

import click
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jinja2 import Environment, FileSystemLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from birdclef.models.embedding.tilenet import TileNet
from birdclef.workflows import motif

ROOT = Path(__file__).parent.parent


@click.group()
def evaluation():
    pass


@evaluation.command()
@click.option(
    "--intra",
    default="brnowl",
    help="Singular species to perform intraclass inspection",
)
@click.option(
    "--inter",
    default="brnowl,skylar,norcar",
    help="Array of 3 species to perform interclass comparison",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("../../data/processed/model/2022-05-17-v7/embedding.ckpt"),
)
@click.option(
    "--parquet",
    type=click.Path(exists=True, dir_okay=False),
    default=Path("../../data/processed/2022-04-03-motif-consolidated.parquet"),
)
@click.option(
    "--outputdir",
    type=click.Path(exists=True, dir_okay=True),
    default=Path(f"{ROOT}/data/intermediate/generated_resources"),
)
@click.option(
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default=Path("../../data/raw/birdclef-2022"),
)
@click.option("--dim", default=512)
@click.option("--name", default=date.today())
def main(intra, inter, checkpoint, parquet, outputdir, root, dim, name):
    model = TileNet.load_from_checkpoint(checkpoint, dim=512)
    df = pd.read_parquet(parquet)
    df["species"] = df.source_name.apply(lambda x: x.split("/")[1])
    df[["species"]].groupby("species").size().sort_values(ascending=False)

    data = []
    for row in df[df.species == intra].sample(50).itertuples():
        y = motif.load_audio(root / row.source_name, int(row.motif_0), 5)
        data.append(y)

    z = data[0]
    plt.plot(z)
    os.makedirs(f"{outputdir}/{name}", exist_ok=True)
    plt.savefig(f"{outputdir}/{name}/{intra}1.png", bbox_inches="tight")
    intrafigures = [f"{outputdir}/{name}/{intra}1.png"]
    plt.clf()

    emb = model(torch.from_numpy(np.array(data))).detach().numpy()

    g = PCA(n_components=2).fit_transform(emb)
    plt.scatter(g[:, 0], g[:, 1])

    plt.savefig(f"{outputdir}/{name}/{intra}2.png", bbox_inches="tight")
    intrafigures.append(f"{outputdir}/{name}/{intra}2.png")
    plt.clf()

    kmeans = KMeans(n_clusters=3).fit(emb)
    kmeans.labels_
    plt.scatter(g[:, 0], g[:, 1], c=kmeans.labels_)
    plt.savefig(f"{outputdir}/{name}/{intra}3.png", bbox_inches="tight")
    intrafigures.append(f"{outputdir}/{name}/{intra}3.png")
    plt.clf()
    print("Created Intraclass Figures")

    # Interclass Clustering and Comparison
    data = []
    labels = []
    for row in df[df.species.isin(inter.split(","))].sample(300).itertuples():
        y = motif.load_audio(root / row.source_name, int(row.motif_0), 5)
        data.append(y)
        labels.append(row.species)

    for i in range(3):
        z = data[i]
        plt.plot(z)
        plt.clf()
        ipd.display(ipd.Audio(z, rate=32000))

    emb = model(torch.from_numpy(np.array(data))).detach().numpy()
    le = LabelEncoder().fit(labels)

    g = PCA(n_components=2).fit_transform(emb)
    plt.scatter(g[:, 0], g[:, 1], c=le.transform(labels))
    store = f"{outputdir}/{name}/{inter}1.png"
    plt.savefig(store, bbox_inches="tight")
    plt.clf()
    print("Generated Interclass Figures")

    # Create html file with Jinja2 template
    file_loader = FileSystemLoader("../../birdclef/workflows/templates")
    env = Environment(loader=file_loader)

    template = env.get_template("index.html.j2")
    output = template.render(
        intraspecies=intra,
        interspecies=inter,
        intrafigures=intrafigures,
        interfigures=store,
    )

    with open(f"{outputdir}/{name}/index.html.j2", "w") as fp:
        fp.write(output)
    webbrowser.open_new(f"{outputdir}/{name}/index.html.j2")


if __name__ == "__main__":
    main()
