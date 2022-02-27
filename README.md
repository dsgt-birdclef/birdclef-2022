# birclef-2022

## quickstart

First we extract the most common motif for every birdcall using simple. We use
chroma energy normalized statistics (CENS) using a rate of 10 samples a second
over a 5 second window.

```bash
python -m birdclef.workflows.motif extract
python -m birdclef.workflows.motif consolidate
```

This generates a new dataset with the location of the motif and it's closest
neighbor. The entire matrix profile is made available for further analysis.
