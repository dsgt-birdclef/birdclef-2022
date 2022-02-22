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
