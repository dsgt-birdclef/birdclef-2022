from birdclef.models.embedding.tilenet import TileNet
import pandas as pd
from birdclef.workflows import motif
from pathlib import Path
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from jinja2 import Environment, FileSystemLoader
import webbrowser
import click
from ast import literal_eval
import os


@click.command()
@click.option('--intra', default="brnowl", help='Singular species to perform intraclass inspection')
@click.option('--inter', default=["brnowl", "skylar", "norcar"], help='Array of 3 species to perform interclass comparison')
def main(intra, inter):
    model = TileNet.load_from_checkpoint(
        "../data/processed/model/2022-04-12-v4/"
        "embedding.ckpt",
        z_dim=64,
    )

    df = pd.read_parquet("../data/processed/2022-04-03-motif-consolidated.parquet")
    df["species"] = df.source_name.apply(lambda x: x.split("/")[1])
    df[["species"]].groupby("species").size().sort_values(ascending=False)

    data = []
    root = Path("../data/raw/birdclef-2022")
    for row in df[df.species == intra].sample(50).itertuples():
        y = motif.load_audio(root / row.source_name, int(row.motif_0), 5)
        data.append(y)

    z = data[0]
    plt.plot(z)
    os.makedirs('generated_resources', exist_ok=True)
    plt.savefig('generated_resources/' + intra + '1.png', bbox_inches='tight')
    intrafigures = ['generated_resources/' + intra + '1.png']
    plt.clf()
    intraAudio1 = ipd.Audio(z, rate=32000)

    np.array(data).shape

    emb = model(torch.from_numpy(np.array(data))).detach().numpy()

    g = PCA(n_components=2).fit_transform(emb)
    plt.scatter(g[:, 0], g[:, 1])
    plt.savefig('generated_resources/' + intra + '2.png', bbox_inches='tight')
    intrafigures.append('generated_resources/' + intra + '2.png')
    plt.clf()

    kmeans = KMeans(n_clusters=3).fit(emb)
    kmeans.labels_
    plt.scatter(g[:, 0], g[:, 1], c=kmeans.labels_)
    plt.savefig('generated_resources/' + intra + '3.png', bbox_inches='tight')
    intrafigures.append('generated_resources/' + intra + '3.png')
    plt.clf()

    # Interclass Clustering and Comparison
    data = []
    root = Path("../data/raw/birdclef-2022")
    labels = []
    for row in df[df.species.isin(literal_eval(inter))].sample(300).itertuples():
        y = motif.load_audio(root / row.source_name, int(row.motif_0), 5)
        data.append(y)
        labels.append(row.species)

    for i in range(3):
        # print(f"{i} class {labels[i]}")
        z = data[i]
        plt.plot(z)
        plt.clf()
        ipd.display(ipd.Audio(z, rate=32000))

    emb = model(torch.from_numpy(np.array(data))).detach().numpy()
    le = LabelEncoder().fit(labels)

    g = PCA(n_components=2).fit_transform(emb)
    plt.scatter(g[:, 0], g[:, 1], c=le.transform(labels))
    plt.savefig('generated_resources/inter1.png', bbox_inches='tight')
    plt.clf()

    # Create html file with Jinja2 template
    file_loader = FileSystemLoader('templates')
    env = Environment(loader=file_loader)

    template = env.get_template('index.html')
    output = template.render(intraspecies = intra, interspecies = inter, intrafigures = intrafigures, interfigures = 'generated_resources/inter1.png')

    with open('index.html', 'w') as fp:
        fp.write(output)
    webbrowser.open_new('index.html')


if __name__ == "__main__":
    main()
