import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierNet(pl.LightningModule):
    def __init__(self, z_dim: int, n_classes: int):
        super(ClassifierNet, self).__init__()
        self.lr = 1e-4

        # NOTE: why is this example for the raw audio and not of the embedding?
        # it might be because the datamodule installs a callback to transform
        # data after moving the data over to the device
        self.example_input_array = (
            torch.rand(7, 5 * 32000).float(),
            torch.rand(7, n_classes).float(),
        )

        self.layer1 = nn.Linear(z_dim, 128)
        self.layer2 = nn.Linear(128, n_classes)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def encode(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def forward(self, x, *args):
        return self.encode(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def _step_losses(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step_losses(batch, batch_idx)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step_losses(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step_losses(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def pred_step(self, batch, batch_idx):
        return self.encode(batch)
