import click

from .classify import classify
from .classify_nn import classify_nn
from .embed import embed
from .evaluation import evaluation
from .label_studio import label_studio
from .motif import motif
from .nocall import nocall


def cli():
    commands = dict(
        classify=classify,
        classify_nn=classify_nn,
        embed=embed,
        label_studio=label_studio,
        motif=motif,
        nocall=nocall,
        evaluation=evaluation,
    )

    @click.group(commands=commands)
    def group():
        pass

    group()


if __name__ == "__main__":
    cli()
