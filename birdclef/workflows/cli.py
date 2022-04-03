import click

from .classify import classify
from .embed import embed
from .label_studio import label_studio
from .motif import motif
from .nocall import nocall


def cli():
    commands = dict(
        classify=classify,
        embed=embed,
        label_studio=label_studio,
        motif=motif,
        nocall=nocall,
    )

    @click.group(commands=commands)
    def group():
        pass

    group()


if __name__ == "__main__":
    cli()
