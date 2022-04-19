# Label Studio

This directory contains code related to label studio. This section is lacking in
documentation. The approach may be unncessary given a good data loading strategy.

## notes

```bash
pipx install label-studio
label-studio

# in a new terminal
cd label-studio
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
