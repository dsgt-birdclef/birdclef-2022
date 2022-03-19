#!/usr/env/bin python
from pathlib import Path
from subprocess import run

import click

root = Path(__file__).parent.parent


@click.group()
def sync():
    pass


@sync.command()
def up():
    """Synchronize files from localhost to remote."""
    src = "data/processed"
    dst = "gs://birdclef-2022/processed"
    run(f"gsutil -m rsync -r {src}/ {dst}/", shell=True, cwd=root)


@sync.command()
def down():
    """Synchronize files from remote to localhost."""
    src = "gs://birdclef-2022/processed"
    dst = "data/processed"
    run(f"gsutil -m rsync -r {src}/ {dst}/", shell=True, cwd=root)


if __name__ == "__main__":
    sync()
