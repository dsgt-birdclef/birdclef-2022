# birclef-2022

## notes

### papers

- Jean, N., Wang, S., Samar, A., Azzari, G., Lobell, D., & Ermon, S. (2019, July). Tile2vec: Unsupervised representation learning for spatially distributed data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 3967-3974). https://arxiv.org/pdf/1805.02855.pdf
- Silva, D. F., Yeh, C. C. M., Batista, G. E., & Keogh, E. J. (2016, August). SiMPle: Assessing Music Similarity Using Subsequences Joins. In ISMIR (pp. 23-29). https://www.cs.ucr.edu/~eamonn/MP_Music_ISMIR.pdf
- Silva, D. F., Yeh, C. C. M., Zhu, Y., Batista, G. E., & Keogh, E. (2018). Fast similarity matrix profile for music analysis and exploration. IEEE Transactions on Multimedia, 21(1), 29-38. https://www.cs.ucr.edu/~eamonn/final-fast-similarity-3.pdf
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.https://arxiv.org/pdf/1710.09412.pdf
- McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015, July). librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference (Vol. 8, pp. 18-25). https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.701.4288&rep=rep1&type=pdf
- Cheuk, K. W., Anderson, H., Agres, K., & Herremans, D. (2020). nnaudio: An on-the-fly gpu audio to spectrogram conversion toolbox using 1d convolutional neural networks. IEEE Access, 8, 161981-162003. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9174990

### motif mining

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

We come back to this in order to generate motif features for our classifier. We
start off a naive approach, where we random sample motifs/discords from across
the entire dataset. This allows us to build from a large set of examples. We
might want to actually limit these sames to the scored species instead, since
those examples may be more indicative of real species. We hope that the
gradient-boosted decision tree will be able to take advantage of distances from
these examples effectively.

```powershell
python -m birdclef.workflows.motif extract `
    --cens-sr 10 `
    --mp-window 30 `
    --sample-k 64 `
    --output 2022-03-18-motif-sample-k-64-v1
```

And just to check that this has a somewhat diverse cast of chirps, we can create an audio file from it.

```powershell
python -m birdclef.workflows.motif motif-track `
    --input data/intermediate/2022-03-18-motif-sample-k-64-v1 `
    data/intermediate/2022-03-18-motif-sample-k-64-v1-motif.wav

python -m birdclef.workflows.motif motif-track `
    --input data/intermediate/2022-03-18-motif-sample-k-64-v1 `
    --index discord_0 `
    data/intermediate/2022-03-18-motif-sample-k-64-v1-discord.wav
```

### triplet formation

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

### training embedding

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

### no-call classifier

After training the embedding, we train a classifier on some soundscapes from 2021. This is already conveniently labeled for us, so we can quickly get up to
speed.

```powershell
python -m birdclef.workflows.nocall fit-soundscape-cv `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_1/checkpoints/epoch=2-step=10872.ckpt `
    data/intermediate/2022-03-09-lgb-test-01.txt

Early stopping, best iteration is: [35]
cv_agg's auc: 0.784049 + 0.0216534
```

However, we find a bug in the training method, so we'll retrain a new embedding.

```powershell
python -m birdclef.workflows.motif extract-triplets `
    data/intermediate/2022-02-26-motif-triplets-5e+05.parquet `
    --output data/intermediate/2022-03-12-motif-triplets-5e+

python -m birdclef.workflows.embed fit `
    .\data\intermediate\2022-02-26-motif-triplets-5e+05.parquet `
    .\data\intermediate\2022-03-12-motif-triplets-5e+05

tensorboard --logdir data/intermediate/embedding

python -m birdclef.workflows.nocall fit-soundscape-cv `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    data/intermediate/2022-03-12-lgb-test-01.txt

Early stopping, best iteration is: [34]
cv_agg's auc: 0.755708 + 0.0165101
```

The results are considerably worse, for some reason. Since we stopped early,
let's continue to train for another day or so to see if this improves a bit.

```powershell
python -m birdclef.workflows.embed fit `
    .\data\intermediate\2022-02-26-motif-triplets-5e+05.parquet `
    .\data\intermediate\2022-03-12-motif-triplets-5e+05 `
    --checkpoint version_2/checkpoints/epoch=2-step=10849.ckpt

python -m birdclef.workflows.nocall fit-soundscape-cv `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_3/checkpoints/epoch=5-step=24488.ckpt `
    data/intermediate/2022-03-12-lgb-test-02.txt

Early stopping, best iteration is: [32]
cv_agg's auc: 0.759764 + 0.0268462
```

It's seriously disappointing that hitting exit early did not capture any of the
work in the 10 iterations since the last checkpoint. And again, the cv scores
are fairly low.

```powershell
python -m birdclef.workflows.nocall fit-soundscape `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    data/intermediate/2022-03-12-lgb.txt
```

### submission classifier

We'll use the primary motif from each track as a training example to train the
final classifier.

```powershell
python -m birdclef.workflows.classify train `
    --birdclef-root data/raw/birdclef-2021 `
    --motif-root data/intermediate/2022-03-12-extracted-primary-motif `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    --filter-set data/raw/birdclef-2022/scored_birds.json `
    data/processed/model/2022-03-18-v1

python -m birdclef.workflows.classify predict `
    --birdclef-root data/raw/birdclef-2022 `
    --classifier-source data/processed/model/2022-03-18-v1 `
    data/processed/submission/2022-03-18-v1.csv
```

We add some modifications to use the sklearn interface.

```powershell
python -m birdclef.workflows.classify train `
    --birdclef-root data/raw/birdclef-2021 `
    --motif-root data/intermediate/2022-03-12-extracted-primary-motif `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    --filter-set data/raw/birdclef-2022/scored_birds.json `
    data/processed/model/2022-03-18-v2

python -m birdclef.workflows.classify predict `
    --birdclef-root data/raw/birdclef-2022 `
    --classifier-source data/processed/model/2022-03-18-v2 `
    data/processed/submission/2022-03-18-v2.csv
```

And now we add motif features:

```powershell
python -m birdclef.workflows.classify train `
    --birdclef-root data/raw/birdclef-2021 `
    --motif-root data/intermediate/2022-03-12-extracted-primary-motif `
    --use-ref-motif `
    --ref-motif-root data/intermediate/2022-03-18-motif-sample-k-64-v1 `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    --filter-set data/raw/birdclef-2022/scored_birds.json `
    data/processed/model/2022-04-02-v3

python -m birdclef.workflows.classify predict `
    --birdclef-root data/raw/birdclef-2022 `
    --classifier-source data/processed/model/2022-04-02-v3 `
    data/processed/submission/2022-04-02-v3.csv
```

We're going to modify the motif extraction be able to take the top motifs (as well as the discord).

```powershell
python -m birdclef.workflows.motif consolidate
python -m birdclef.workflows.motif extract-top-motif `
    --output data/intermediate/2022-04-03-extracted-top-motif

python -m birdclef.workflows.classify prepare-dataset `
    --motif-root data/intermediate/2022-04-03-extracted-top-motif `
    --num-per-class 100 `
    --limit 100 `
    data/intermediate/2022-04-03-train-augment-100

python -m birdclef.workflows.classify prepare-dataset `
    --motif-root data/intermediate/2022-04-03-extracted-top-motif `
    --num-per-class 250 `
    data/intermediate/2022-04-03-train-augment-250

python -m birdclef.workflows.classify prepare-dataset `
    --motif-root data/intermediate/2022-04-03-extracted-top-motif `
    --num-per-class 2500 `
    data/intermediate/2022-04-03-train-augment-2500
```

For now, we also also disable the use of reference motif during training because
otherwise predictions on kaggle is really slow.

```
python -m birdclef.workflows.classify train `
    --birdclef-root data/raw/birdclef-2021 `
    --motif-root data/intermediate/2022-04-03-train-augment-250 `
    --no-use-ref-motif `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    --filter-set data/raw/birdclef-2022/scored_birds.json `
    data/processed/model/2022-04-12-v4

# test score: 0.7278375308887418

python -m birdclef.workflows.classify predict `
    --birdclef-root data/raw/birdclef-2022 `
    --classifier-source data/processed/model/2022-04-12-v4 `
    data/processed/submission/2022-04-12-v4.csv
```

### kaggle submission

See the following two notebooks:

- https://www.kaggle.com/code/acmiyaguchi/model-sync
- https://www.kaggle.com/code/acmiyaguchi/motif-join-and-embedding

The first notebook downloads any models in the shared GCP bucket
(`gs://birdclef-2022`). It also downloads the main package in this repository,
using a private github token.

The second notebook contains the actual code. It simply mounts the output of the
`model-sync` notebook and calls the `birdclef.workflows.classify` command.

### Label Studio

```bash
pipx install label-studio
label-studio

# in a new directory
docker-compose up

python -m birdclef.workflows.label_studio train-list `
    --pattern birdclef-2022/train_audio/skylar/* `
    data/intermediate/studio/skylar.txt
```

Annotating all of the audio files for a single species would be much too time
consuming. Instead, we will instead just annotate motifs from across all
species. We can seed this process by classifying calls into call or no call. By
improving our no-call classifier, we can start to build a much larger repository
of positive examples.

```bash
python -m birdclef.workflows.motif extract-primary-motif

python -m birdclef.workflows.label_studio motif-list `
    --embedding-checkpoint data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt `
    --nocall-params data/intermediate/2022-03-12-lgb.txt `
    data/intermediate/studio/motif.json
```

Labeling is a laborious process, but many of the samples seem to be working
okay.
