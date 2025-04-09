def audio_slicer(item, win_dur, sr, hop_dur=None, drop_last=True):
    audio = item
    N = len(audio)
    win_len = int(sr * win_dur)
    if not drop_last:
        win_len = min(win_len, N)
    hop_len = win_len if hop_dur is None else int(sr * hop_dur)
    for i in range(0, N - win_len + 1, hop_len):
        yield audio[i : i + win_len]


def multi_audio_slicer(items, win_dur, sr, hop_dur=None, drop_last=True):
    for item in items:
        yield from audio_slicer(item, win_dur, sr, hop_dur, drop_last)
