from birdclef.models.embedding import datasets


def test_tile_triplets_triplets_dataset(metadata_df, extract_triplet_path):
    dataset = datasets.TileTripletsDataset(metadata_df, extract_triplet_path)
    count = 0
    for i in range(len(dataset)):
        assert dataset[i]
        count += 1
    assert count == metadata_df.shape[0]


def test_tile_triplets_datamodule(metadata_df, extract_triplet_path):
    dm = datasets.TileTripletsDataModule(
        metadata_df, extract_triplet_path, batch_size=1
    )
    dm.setup()
    assert len(dm.train_dataloader()) == 8
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1
