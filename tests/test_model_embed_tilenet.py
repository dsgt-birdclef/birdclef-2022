import pytorch_lightning as pl

from birdclef.models.embedding import datasets, tilenet


def test_tilenet_train(metadata_df, tile_path):
    data_module = datasets.TileTripletsDataModule(metadata_df, tile_path, batch_size=1)
    model = tilenet.TileNet(z_dim=512, n_mels=64)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, data_module)
