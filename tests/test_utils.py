import numpy as np
import soundfile as sf
import pytest
from birdclef.utils import load_audio


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


@pytest.mark.parametrize("offset", [0, -1, 9])
def test_load_audio(tone_long, sr, offset):
    duration = 5
    length = sr * duration
    y = load_audio(tone_long, offset, duration=duration)
    assert y.shape[0] == length
    assert (y == 0).sum() == 0


def test_load_audio_short_centered(tone_short, sr):
    duration = 5
    length = sr * duration
    y = load_audio(tone_short, 0, duration=duration)
    assert y.shape[0] == length
    assert (y == 0.0).sum() > 0
