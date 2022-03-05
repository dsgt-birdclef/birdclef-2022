# birclef-2022

## quickstart

First we extract the most common motif for every birdcall using simple. We use
chroma energy normalized statistics (CENS) using a rate of 10 samples a second
over a 5 second window.

```powershell
python -m birdclef.workflows.motif extract
python -m birdclef.workflows.motif consolidate
python -m birdclef.workflows.motif generate-triplets --samples 10000
python -m birdclef.workflows.motif generate-triplets --samples 1000
python -m birdclef.workflows.motif extract-triplets `
    data/intermediate/2022-02-26-motif-triplets-1e+03.parquet
python -m birdclef.workflows.motif extract-triplets `
    data/intermediate/2022-02-26-motif-triplets-1e+05.parquet `
    --output data/intermediate/2022-03-05-motif-triplets-1e+05
```

This generates a new dataset with the location of the motif and it's closest
neighbor. The entire matrix profile is made available for further analysis. We
extract the motifs out into npy files because it is significantly slower to read
the samples on the fly. The resulting size of the dataset is on the order of
65GB, for a total of 100k samples.

The sampling methodology splits the dataset into four partitions.

For neighbors:

- motif with self
- motif with motif in neighboring species

For the non-neighbor:

- random motif in non-neighboring species
- random clip in non-neighboring species

- https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html
- https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

```powershell
python -m birdclef.workflows.embed fit `
    .\data\intermediate\2022-02-26-motif-triplets-1e+03.parquet `
    .\data\intermediate\2022-03-02-extracted-triplets

python -m birdclef.workflows.embed fit `
    .\data\intermediate\2022-02-26-motif-triplets-1e+05.parquet `
    .\data\intermediate\2022-03-05-motif-triplets-1e+05
```

Here, we do some commenting in the script to figure out what the optimal batch
size will be (turns out we can't fit more than 64 rows into memory) and what the
learning rate should be.
