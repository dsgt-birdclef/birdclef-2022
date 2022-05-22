import numpy as np
import pytorch_lightning as pl
import torch
from torchsummary import summary

from birdclef.models.classifier_nn.datasets import ClassifierDataModule
from birdclef.models.classifier_nn.model import ClassifierNet


def test_model_train(train_root, label_encoder, model_checkpoint, z_dim):
    dm = ClassifierDataModule(
        train_root, label_encoder, model_checkpoint, z_dim, batch_size=5, num_workers=1
    )
    model = ClassifierNet(z_dim=z_dim, n_classes=len(label_encoder.classes_))
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dm)

    metrics = trainer.callback_metrics
    print(metrics)
    assert np.abs(metrics["loss"].detach()) > 0
    summary(model, torch.randn((5, z_dim)))


def test_model_batch_gradient_does_not_mix(label_encoder, z_dim):
    model = ClassifierNet(z_dim=z_dim, n_classes=len(label_encoder.classes_))
    n = 0
    example_input = torch.randn((5, z_dim)).to(model.device)
    example_input.requires_grad = True

    model.zero_grad()
    output = model(example_input)
    output[n].abs().sum().backward()

    zero_grad_inds = list(range(example_input.size(0)))
    zero_grad_inds.pop(n)

    assert example_input.grad[zero_grad_inds].abs().sum().item() == 0
