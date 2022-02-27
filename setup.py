from distutils.core import setup

setup(
    name="birdclef",
    version="0.1.1",
    description="Utilities for birdclef",
    author="Anthony Miyaguchi",
    author_email="acmiyaguchi@gmail.com",
    url="https://github.com/acmiyaguchi/birdclef-2022",
    packages=["birdclef"],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyspark",
        "librosa",
        "click",
        "tqdm",
        "pyarrow",
    ],
)
