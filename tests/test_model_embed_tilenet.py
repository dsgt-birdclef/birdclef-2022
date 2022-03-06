import pytest
import pytorch_lightning as pl
from torchsummary import summary
import numpy as np

from birdclef.models.embedding import datasets, tilenet


@pytest.mark.parametrize("z_dim", [64, 113])
def test_tilenet_train(metadata_df, extract_triplet_path, z_dim):
    # test that the model actually runs
    data_module = datasets.TileTripletsDataModule(
        metadata_df,
        extract_triplet_path,
        batch_size=1,
        num_workers=1,
    )
    model = tilenet.TileNet(z_dim=z_dim, n_mels=64)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, data_module)

    metrics = trainer.callback_metrics
    print(metrics)
    assert np.abs(metrics["loss"].detach()) > 0

    # assert the shape of the data. Since we're predicting, we'll only keep the
    # anchor from the batch
    for batch in data_module.val_dataloader():
        x = batch["anchor"]
        yhat = model(x)
    # print out the summary from the last batch
    summary(model, x)
    assert yhat.shape[1] == z_dim
