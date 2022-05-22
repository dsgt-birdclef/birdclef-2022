import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from birdclef.models.classifier_nn.datasets import (
    ClassifierSimpleDataset,
    ToEmbedSpace,
    ToFloatTensor,
)
from birdclef.models.classifier_nn.model import ClassifierNet


def predict(bird, train_audio, filter_set, classify_root):
    paths = list(train_audio.glob(f"{bird}/*.ogg"))
    metadata = json.loads((classify_root / "metadata.json").read_text())
    label_encoder = LabelEncoder()
    label_encoder.fit(["noise"] + filter_set)

    dataset = ClassifierSimpleDataset(
        paths[:1],
        label_encoder,
        transform=transforms.Compose(
            [
                ToFloatTensor(),
                ToEmbedSpace(
                    classify_root / "embedding.ckpt", z_dim=metadata["embedding_dim"]
                ),
            ]
        ),
    )
    model = ClassifierNet.load_from_checkpoint(
        classify_root / "classify.ckpt",
        z_dim=metadata["embedding_dim"],
        n_classes=len(label_encoder.classes_),
    )

    pred = None
    actual = None
    for X, y in DataLoader(dataset, batch_size=32):
        y_pred = model.to(X.device)(X).squeeze(1).detach().numpy()
        y = y.detach().numpy()
        print(y_pred.shape, y.shape)
        pred = np.vstack([pred, y_pred]) if pred is not None else y_pred
        actual = np.vstack([actual, y]) if actual is not None else y
    return pred, actual


def plot(bird, pred, actual):
    plt.title(f"average prediction for each label ({bird})")
    plt.plot(actual.mean(axis=0), label="actual")
    plt.plot(pred.mean(axis=0), label="predicted")
    plt.legend()
    plt.show()
