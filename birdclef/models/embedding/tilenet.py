"""Modified ResNet-18 in PyTorch.
Reference:
[0] Jean, N., Wang, S., Samar, A., Azzari, G., Lobell, D., & Ermon, S. (2019,
    July). Tile2vec: Unsupervised representation learning for spatially
    distributed data. In Proceedings of the AAAI Conference on Artificial
    Intelligence (Vol. 33, No. 01, pp. 3967-3974).
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Cheuk, K. W., Anderson, H., Agres, K., & Herremans, D. (2020). nnaudio: An
    on-the-fly gpu audio to spectrogram conversion toolbox using 1d
    convolutional neural networks. IEEE Access, 8, 161981-162003.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features.mel import MelSpectrogram
from torch.autograd import Variable


class ResidualBlock(pl.LightningModule):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(pl.LightningModule):
    def __init__(
        self,
        z_dim=512,
        n_mels=64,
        num_blocks=[2, 2, 2, 2, 2],
        hop_length=512,
        fmin=0,
        fmax=16000,
        sample_rate=22050,
    ):
        super(TileNet, self).__init__()
        self.z_dim = z_dim
        self.in_planes = 64

        self.spec_layer = MelSpectrogram(
            n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax, sr=sample_rate
        )
        print(self.spec_layer)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4], stride=2)
        self.fc1 = nn.Linear(self.z_dim * 3, self.z_dim)

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.spec_layer(x)
        x = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        # for some reason, we get 3x the size of the final dimension
        z = self.fc1(z)
        return z

    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = -torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (
            self.encode(patch),
            self.encode(neighbor),
            self.encode(distant),
        )
        return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        p, n, d = (
            Variable(batch["anchor"]),
            Variable(batch["neighbor"]),
            Variable(batch["distant"]),
        )
        loss, l_n, l_d, l_nd = self.loss(p, n, d, margin=1, l2=0)
        return {
            "loss": loss,
            "loss_n": l_n.detach(),
            "loss_d": l_d.detach(),
            "loss_nd": l_nd.detach(),
        }

    def validation_step(self, batch, batch_idx):
        # NOTE: this is (probably) correct, since the training step itself
        # shouldn't be calling the backward step.
        losses = self.training_step(batch, batch_idx)
        for key, value in losses.items():
            self.log(key, value)

    def test_step(self, batch, batch_idx):
        losses = self.training_step(batch, batch_idx)
        for key, value in losses.items():
            self.log(key, value)

    def pred_step(self, batch, batch_idx):
        return self.encode(batch)
