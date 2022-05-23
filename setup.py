import setuptools

setuptools.setup(
    name="birdclef",
    version="0.13.0",
    description="Utilities for birdclef",
    author="Anthony Miyaguchi",
    author_email="acmiyaguchi@gmail.com",
    url="https://github.com/acmiyaguchi/birdclef-2022",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "librosa",
        "soundfile",
        "click",
        "tqdm",
        "pyarrow",
        "torch",
        "nnAudio",
        "pytorch-lightning",
        "torch-audiomentations",
        "torch-summary",
        "audiomentations",
        "lightgbm",
        "simple-fast-python",
        'importlib-metadata>=0.12;python_version<"3.8"',
    ],
    entry_points="""
        [console_scripts]
        birdclef=birdclef.workflows.cli:cli
    """,
)
