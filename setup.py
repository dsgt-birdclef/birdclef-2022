import setuptools

setuptools.setup(
    name="birdclef",
    version="0.6.0",
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
        "torch-summary",
        "lightgbm",
    ],
)
