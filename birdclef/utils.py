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
    """Load an audio fragment from a file.

    This is typically used for padding smaller audio files with various offsets.
    """
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
    # offset is less than 0, we do some negative padding
    elif offset <= 0:
        offset = pad_size + max(offset, -pad_size)
    elif offset + length >= y_pad.shape[0]:
        offset = y_pad.shape[0] - length
    else:
        # we can start at the normal offset location
        offset += pad_size

    y_trunc = y_pad[offset : offset + length]
    return np.resize(np.moveaxis(y_trunc, -1, 0), length)


def slice_seconds(
    data, sample_rate, seconds=5, pad_seconds=0, step=None, padding_type="center"
):
    # return 2d array of the original data
    valid_padding_types = ["center", "right", "right-align"]
    if padding_type not in valid_padding_types:
        raise ValueError(f"padding_type must be one of: {valid_padding_types}")

    # compute step size
    k = sample_rate * seconds
    pad = sample_rate * pad_seconds
    step = k + pad if step is None else int(sample_rate * step)

    remainder = len(data) % step
    if remainder:
        # pad dataset based on padding type
        padding_size = step - remainder
        if padding_type == "right":
            data = np.pad(data, (0, padding_size))
        elif padding_type == "center":
            left_padding = padding_size // 2
            right_padding = left_padding if padding_size % 2 == 0 else left_padding + 1
            data = np.pad(data, (left_padding, right_padding))
        elif padding_type == "right-align":
            # Get a full slice from the right side, then truncate the data
            # to divide evenly by step size.  Append the full slice to the end.
            last_slice = data[-step:]
            remaining_data = data[: (len(data) // step * step)]
            data = np.hstack((remaining_data, last_slice))

    n = len(data)
    indexes = np.array(
        [np.arange(i, i + k + pad) for i in range(0, n, step) if i + k + pad <= n]
    ).astype(int)
    indexed = data[indexes]
    if indexed.shape[0] == 0:
        return []

    time_index = np.arange(indexed.shape[0] + 1) * seconds
    return list(zip(time_index, indexed))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.

    https://stackoverflow.com/a/312464
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def transform_input(model, device, X, batch_size=50):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    res = []
    for batch in dataloader:
        # note that we can't use the trainer because the batches end up being lists
        res.append(model(batch[0].to(device)).cpu().detach().numpy())
    return np.concatenate(res)
