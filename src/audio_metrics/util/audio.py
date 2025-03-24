def audio_slicer(item, win_dur, sr, hop_dur=None):
    audio = item
    N = len(audio)
    win_len = int(sr * win_dur)
    hop_len = win_len if hop_dur is None else int(sr * hop_dur)
    for i in range(0, N - win_len + 1, hop_len):
        yield audio[i : i + win_len]


def multi_audio_slicer(items, win_dur, sr, hop_dur=None, drop_last=True):
    if not drop_last:
        raise NotImplementedError
    for item in items:
        yield from audio_slicer(item, win_dur, sr, hop_dur)
