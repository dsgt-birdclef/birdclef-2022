# birclef-2022

## quickstart

First we extract the most common motif for every birdcall using simple. We use
chroma energy normalized statistics (CENS) using a rate of 10 samples a second
over a 5 second window.

```bash
python -m birdclef.workflows.motif extract
python -m birdclef.workflows.motif consolidate
python -m birdclef.workflows.motif generate-triplets --samples 10000
python -m birdclef.workflows.motif generate-triplets --samples 1000
```

This generates a new dataset with the location of the motif and it's closest
neighbor. The entire matrix profile is made available for further analysis.

- https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html
- https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

```bash
python -m birdclef.workflows.embed fit .\data\intermediate\2022-02-26-motif-triplets-1e+05.parquet
```
