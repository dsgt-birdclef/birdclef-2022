#!/usr/env/bin python
from pathlib import Path
from subprocess import run

import click

root = Path(__file__).parent.parent

PATHS = [
    ("data/processed", "gs://birdclef-2022/processed"),
    # NOTE: only includes soundscapes
    ("data/raw/birdclef-2021", "gs://birdclef-2022/raw/birdclef-2021"),
    ("data/raw/birdclef-2022", "gs://birdclef-2022/raw/birdclef-2022"),
]


@click.group()
def sync():
    pass


@sync.command()
def up():
    """Synchronize files from localhost to remote."""
    for src, dst in PATHS:
        run(f"gsutil -m rsync -r {src}/ {dst}/", shell=True, cwd=root)


@sync.command()
def down():
    """Synchronize files from remote to localhost."""
    for dst, src in PATHS:
        Path(dst).mkdir(parents=True, exist_ok=True)
        run(f"gsutil -m rsync -r {src}/ {dst}/", shell=True, cwd=root)


if __name__ == "__main__":
    sync()
