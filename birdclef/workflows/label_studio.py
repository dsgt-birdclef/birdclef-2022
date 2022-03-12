from pathlib import Path

import click


@click.group()
def label_studio():
    pass


@label_studio.command()
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--prefix", type=str, default="http://localhost:8000")
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=False),
    default=Path("data/raw"),
)
@click.option("--pattern", type=str, default="birdclef-2022/train_audio/**/*.ogg")
def train_list(output, prefix, input, pattern):
    files = list(Path(input).glob(pattern))
    Path(output).write_text(
        "\n".join([f"{prefix}/{file.relative_to(input).as_posix()}" for file in files])
    )


if __name__ == "__main__":
    label_studio()
