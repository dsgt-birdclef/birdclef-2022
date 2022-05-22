import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf

from birdclef.utils import load_audio, slice_seconds


@pytest.fixture
def sr():
    return 32000


@pytest.fixture
def tone_long(tmp_path, sr):
    path = tmp_path / "tone.ogg"
    sf.write(path, np.ones(sr * 10) * 3.0, sr, format="ogg", subtype="vorbis")
    yield path


@pytest.fixture
def tone_short(tmp_path):
    sr = 32000
    path = tmp_path / "tone.ogg"
    sf.write(path, np.ones(sr * 3) * 3.0, sr, format="ogg", subtype="vorbis")
    yield path


@pytest.mark.parametrize("offset", [0, -1, 6])
def test_load_audio(tone_long, sr, offset):
    duration = 5
    length = sr * duration
    y = load_audio(tone_long, offset, duration=duration)
    assert y.shape[0] == length
    if offset == 0:
        assert (y == 0).sum() == 0
    else:
        assert (y == 0).sum() == sr


def test_load_audio_short_centered(tone_short, sr):
    duration = 5
    length = sr * duration
    y = load_audio(tone_short, 0, duration=duration)
    assert y.shape[0] == length
    assert (y == 0.0).sum() > 0
    assert (y[:sr] > 0).sum() == 0
    assert (y[-sr:] > 0).sum() == 0


def test_slice_seconds_right_pad():
    x = np.ones(16)
    res = slice_seconds(x, 1, 5, padding_type="right")
    assert len(res) == 4

    # assert 1st slice is correct
    i, v = res[0]
    assert i == 0
    assert (v - np.ones(5)).sum() == 0

    # assert last slice is right-padded correctly
    i, v = res[3]
    assert i == 15
    assert (v - np.array([1, 0, 0, 0, 0])).sum() == 0


def test_slice_seconds_center_pad():
    x = np.ones(17)
    res = slice_seconds(x, 1, 5, padding_type="center")
    assert len(res) == 4

    # assert 1st slice is correct
    i, v = res[0]
    assert i == 0
    assert (v - np.array([0, 1, 1, 1, 1])).sum() == 0

    # assert last slice is center-padded correctly
    i, v = res[3]
    assert i == 15
    assert (v - np.array([1, 1, 1, 0, 0])).sum() == 0


def test_slice_seconds_right_align_pad():
    x = np.arange(1, 8)
    res = slice_seconds(x, 1, 5, padding_type="right-align")
    assert len(res) == 2

    # assert 1st slice is correct
    i, v = res[0]
    assert i == 0
    assert (v - np.arange(1, 6)).sum() == 0

    # assert last slice correct (which will contain some duplication)
    i, v = res[1]
    assert i == 5
    assert (v - np.arange(3, 8)).sum() == 0


def test_slice_seconds_step_size():
    x = np.arange(16)
    res = slice_seconds(x, 1, 4, step=2, padding_type="right")
    print(res)
    assert len(res) == 7
