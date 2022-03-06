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
python -m birdclef.workflows.motif generate-triplets --samples 5e5

python -m birdclef.workflows.motif extract-triplets `
    data/intermediate/2022-02-26-motif-triplets-1e+03.parquet
python -m birdclef.workflows.motif extract-triplets `
    data/intermediate/2022-02-26-motif-triplets-1e+05.parquet `
    --output data/intermediate/2022-03-05-motif-triplets-1e+05

python -m birdclef.workflows.motif generate-triplets --samples 5e5
python -m birdclef.workflows.motif extract-triplets `
    data/intermediate/2022-02-26-motif-triplets-5e+05.parquet `
    --output data/intermediate/2022-03-05-motif-triplets-5e+05
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

We make some distributional assumptions; we want to have have roughly the same
number of samples per class. This does not reflect the frequency of the calls in
the wild, but it should serve to differentiate between the different calls in
the embedding space.

After some development, we chose to further increase our main training dataset
from 100k samples to 500k samples (150gb of data). The downside of this is that
a single epoch will take around 5 hours -- we choose instead to fix the number
of batches per training round so that it takes an hour per epoch. We lose the
semantics of an epoch, which is an entire round over the training dataset.
However, with the augmentation scheme proposed in tile2vec, as well as the
uneven distribution of motifs within the training dataset, we argue that the
semantics are not well founded in the first place. We take the larger 500k
sample dataset, shuffle, and take the first 100k samples. This should have
roughly the same distributional assumptions as the premise, while going through
a larger number of permutations. Note that that for each sample, we also shift
the audio track in time.

- https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html
- https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

```powershell
python -m birdclef.workflows.embed fit `
    .\data\intermediate\2022-02-26-motif-triplets-1e+03.parquet `
    .\data\intermediate\2022-03-02-extracted-triplets

python -m birdclef.workflows.embed fit `
    .\data\intermediate\2022-02-26-motif-triplets-5e+05.parquet `
    .\data\intermediate\2022-03-05-motif-triplets-5e+05
```

Here, we do some commenting in the script to figure out what the optimal batch
size will be (turns out we can't fit more than 64 rows into memory) and what the
learning rate should be. We are unable to utilize 16-bit mixed precision,
because the learning rate tuner ends up note being able to compute the relevant
gradient. This is worth further exploration in the future, because mixed
precision can be upwards to 4x faster than using single precision.

We try a couple of different things here to debug, the first and fore-most was
to introduce a testing procedure to check that the gradient is being computed
with respect to a single entry in a batch. However, it turns out that the entire
batch is used to compute the gradient, and fails. We also tried adding a small
amount of noise to the input signal before it is passed into the the spectrogram
layer. We choose to stick with single precision layers (32 bit floats) due to
numerical stability. Despite this choice, we still see the loss turn into NaN on
occassion.
