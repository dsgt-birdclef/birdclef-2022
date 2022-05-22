import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from birdclef.models.classifier_nn.datasets import (
    ClassifierSimpleDataset,
    ToEmbedSpace,
    ToFloatTensor,
)
from birdclef.models.classifier_nn.model import ClassifierNet


def predict(bird, train_audio, filter_set, classify_root, batch_size=64):
    paths = list(train_audio.glob(f"{bird}/*.ogg"))
    metadata = json.loads((classify_root / "metadata.json").read_text())
    label_encoder = LabelEncoder()
    label_encoder.fit(["noise"] + filter_set)

    # NOTE: if we try to put things on the gpu with some parallelism on the
    # dataloader, things get weird inside the notebook. We can just put
    # everything on the CPU and it works okay. We'll get errors related to
    # serialization of lambdas in the model and whatnot otherwise.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_embed = ToEmbedSpace(
        classify_root / "embedding.ckpt", z_dim=metadata["embedding_dim"]
    )

    dataset = ClassifierSimpleDataset(
        paths,
        label_encoder,
        transform=transforms.Compose([ToFloatTensor(device=device), to_embed]),
    )
    model = ClassifierNet.load_from_checkpoint(
        classify_root / "classify.ckpt",
        z_dim=metadata["embedding_dim"],
        n_classes=len(label_encoder.classes_),
    )

    pred = None
    actual = None
    for X, y in DataLoader(dataset, batch_size=batch_size):
        y_pred = model.to(device)(X).squeeze(1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        pred = np.vstack([pred, y_pred]) if pred is not None else y_pred
        actual = np.vstack([actual, y]) if actual is not None else y
    return pred, actual


def plot(bird, pred, actual):
    index = mode(actual.argmax(axis=1))[0][0]

    plt.title(f"average prediction for each label ({bird}={index}, n={pred.shape[0]})")
    plt.plot(actual.mean(axis=0), label="actual")
    plt.plot(pred.mean(axis=0), label="predicted")
    plt.legend()
    plt.show()

    plt.title(
        f"distribution of predictions for label ({bird}={index}, n={pred.shape[0]})"
    )
    plt.hist(pred[:, index])
    plt.show()
