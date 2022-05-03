# birclef-2022

This repository contains code for the [birdclef-2022] Kaggle competition.

[birdclef-2022]: https://www.kaggle.com/c/birdclef-2022/overview

## quickstart

### repo and data preparation

Checkout the repository to your local machine, and download the data from the
competition website. Ensure the data is extracted to the
`data/raw/birdclef-2022` directory.

```powershell
git clone https://github.com/acmiyaguchi/birdclef-2022
cd birdclef-2022

# download the data to the data/raw directory and extract
mkdir -p data/raw
# ...

# ensure that you can run the following command from the project root
cat data/raw/birdclef-2022/scored_birds.json | wc -l
# 23
```

Install the [Google Cloud SDK][gcp-sdk] and ask for permission to the
`birdclef-2022` bucket. Run the following command to ensure you have the correct
permissions.

```powershell
gsutil cat gs://birdclef-2022/processed/model/2022-04-12-v4/metadata.json

{
  "embedding_source": "data/intermediate/embedding/tile2vec-v2/version_2/checkpoints/epoch=2-step=10849.ckpt",
  "embedding_dim": 64,
  "created": "2022-04-12T23:09:51.920185",
  "cens_sr": 10,
  "mp_window": 20,
  "use_ref_motif": false
}
```

Run the `sync.py` script to pull data down from the remote bucket.

```powershell
python scripts/sync.py down
```

In particular, this will synchronize shared files from the `data/processed`
directory.

[gcp-sdk](https://cloud.google.com/sdk/docs/install-sdk)

## notes

## development

The majority of development notes can be found under the [notes](notes)
directory.

- [acmiyaguchi's notes](notes/acmiyaguchi-NOTES.md)

### kaggle submission

See the following two notebooks:

- https://www.kaggle.com/code/acmiyaguchi/model-sync
- https://www.kaggle.com/code/acmiyaguchi/motif-join-and-embedding

The first notebook downloads any models in the shared GCP bucket
(`gs://birdclef-2022`). It also downloads the main package in this repository,
using a private github token.

The second notebook contains the actual code. It simply mounts the output of the
`model-sync` notebook and calls the `birdclef.workflows.classify` command.

### papers

The approach for this year's competition is focused on unsupervised methods. In
particular, the fast similarity matrix profile and tile2vec papers provide the
technical foundation for methods found in the repository.

- Conde, M. V., Shubham, K., Agnihotri, P., Movva, N. D., & Bessenyei, S.
  (n.d.). Weakly-Supervised Classification and Detection of Bird Sounds in the
  Wild. A BirdCLEF 2021 Solution. 12. http://ceur-ws.org/Vol-2936/paper-123.pdf
- Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning
  solution for avian diversity monitoring. Ecological Informatics, 61, 101236.
  https://www.sciencedirect.com/science/article/pii/S1574954121000273
- Jean, N., Wang, S., Samar, A., Azzari, G., Lobell, D., & Ermon, S. (2019,
  July). Tile2vec: Unsupervised representation learning for spatially
  distributed data. In Proceedings of the AAAI Conference on Artificial
  Intelligence (Vol. 33, No. 01, pp. 3967-3974).
  https://arxiv.org/pdf/1805.02855.pdf
- Silva, D. F., Yeh, C. C. M., Batista, G. E., & Keogh, E. J. (2016, August).
  SiMPle: Assessing Music Similarity Using Subsequences Joins. In ISMIR (pp.
  23-29). https://www.cs.ucr.edu/~eamonn/MP_Music_ISMIR.pdf
- Silva, D. F., Yeh, C. C. M., Zhu, Y., Batista, G. E., & Keogh, E. (2018). Fast
  similarity matrix profile for music analysis and exploration. IEEE
  Transactions on Multimedia, 21(1), 29-38.
  https://www.cs.ucr.edu/~eamonn/final-fast-similarity-3.pdf
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond
  empirical risk minimization. arXiv preprint
  arXiv:1710.09412.https://arxiv.org/pdf/1710.09412.pdf
- Cheuk, K. W., Anderson, H., Agres, K., & Herremans, D. (2020). nnaudio: An
  on-the-fly gpu audio to spectrogram conversion toolbox using 1d convolutional
  neural networks. IEEE Access, 8, 161981-162003.
  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9174990
