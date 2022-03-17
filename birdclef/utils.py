from pathlib import Path

import librosa
import numpy as np
import torch


def cens_per_sec(sample_rate, target):
    """Ensure this value is a multiple of 2**6"""
    return (sample_rate // (target * (2**6))) * (2**6)


def compute_offset(index, window_size, cens_total, data_total, window_extra=0):
    """Get the offsets into the original sampled audio by computing the relative
    percentage into the track.

    index: the index into the matrix profile
    window_size: the number of frames used by the matrix profile
    cens_total: the total number of cens frames
    data_total: the total number of audio samples
    window_extra: the number of extra frames to collect from the window
    """
    start = index / (cens_total + window_size)
    end = (index + window_size + window_extra) / (cens_total + window_size)
    offset = (window_size / 2) / (cens_total + window_size)
    return int((start + offset) * data_total), int((end + offset) * data_total)


def load_audio(input_path: Path, offset: float, duration: int = 7, sr: int = 32000):
    # offset is the center point
    y, sr = librosa.load(input_path.as_posix(), sr=sr)
    pad_size = int(np.ceil(sr * duration / 2))
    y_pad = np.pad(y, ((pad_size, pad_size)), "constant", constant_values=0)

    # now get offsets relative to the sample rate
    length = duration * sr
    offset = int(offset * sr)

    # check for left, mid, and right conditions
    if y.shape[0] < length:
        offset = (y_pad.shape[0] // 2) - pad_size
    elif offset <= 0:
        offset = pad_size
    elif offset + length >= y.shape[0]:
        offset = y.shape[0] - pad_size - length
    else:
        pass

    y_trunc = y_pad[offset : offset + length]
    return np.resize(np.moveaxis(y_trunc, -1, 0), length)


def transform_input(model, device, X, batch_size=50):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    res = []
    for batch in dataloader:
        # note that we can't use the trainer because the batches end up being lists
        res.append(model(batch[0].to(device)).cpu().detach().numpy())
    return np.concatenate(res)
