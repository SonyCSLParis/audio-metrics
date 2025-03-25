#!/usr/bin/env python
import argparse
import numpy as np
from time import perf_counter
import pyloudnorm as pyln

from audio_metrics.mix_functions import Meter


def main():
    parser = argparse.ArgumentParser(description="Do something")
    # parser.add_argument("file", help=("File"))
    # args = parser.parse_args()
    sr = 48000
    audio = 0.1 * np.random.random((int(sr * 10 * 60),)) - 0.5
    start, end = sr, int(3 * sr)
    # audio[start:end, :] = 0
    # start, end = 10 * sr, int(15 * sr)
    # audio[start:end, :] = 0
    # audio = audio.astype(np.float)
    meter = Meter(sr)
    lufs = meter.integrated_loudness(audio)
    print("start")
    t0 = perf_counter()
    lufs_fast = meter.integrated_loudness_fast(audio)
    t1 = perf_counter()
    delta_fast = t1 - t0
    print(f"delta fast: {delta_fast:.5f}s")
    t0 = perf_counter()
    lufs_orig = meter.integrated_loudness(audio)
    t1 = perf_counter()
    delta_orig = t1 - t0
    print(f"delta orig: {delta_orig:.5f}s")
    # print speedup:
    speedup = delta_orig / delta_fast
    print(f"Speedup: {speedup:.2f}x")

    print(lufs_fast)
    print(lufs_orig)


if __name__ == "__main__":
    main()
