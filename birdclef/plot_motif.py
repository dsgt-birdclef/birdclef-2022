from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from simple import simple_fast

from birdclef.utils import cens_per_sec


def plot_melspectrogram(ax, path: Path, hop_length=80, mp_window=80 * 5):
    data, sample_rate = librosa.load(path)
    S = librosa.feature.melspectrogram(
        y=data, sr=sample_rate, n_fft=2048, hop_length=80, n_mels=16
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    mp, _ = simple_fast(S, S, mp_window)

    librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=hop_length,
        ax=ax[0],
    )
    ax[0].set(title="Mel-frequency spectrogram")

    # plot the matrix profile mp on the second axis
    ax[1].plot(mp)
    ax[1].set(
        title="Similarity matrix profile (SiMPle)",
        ylabel="distance to nearest neighbor",
    )
    ax[1].set_xticklabels([])
    plt.tight_layout()


def plot_cens(ax, path: Path, cens_sr=10, mp_window=50):
    data, sample_rate = librosa.load(path)

    hop_length = cens_per_sec(sample_rate, cens_sr)
    S = librosa.feature.chroma_cens(y=data, sr=sample_rate, hop_length=hop_length)
    mp, _ = simple_fast(S, S, mp_window)

    librosa.display.specshow(
        S, y_axis="chroma", x_axis="time", hop_length=hop_length, ax=ax[0]
    )
    ax[0].set(title="Chroma energy normalized (CENS) spectrogram")

    # plot the matrix profile mp on the second axis
    ax[1].plot(mp)
    ax[1].set(
        title="Similarity matrix profile (SiMPle)",
        ylabel="distance to nearest neighbor",
    )
    ax[1].set_xticklabels([])
    plt.tight_layout()
