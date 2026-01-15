"""Benchmark and profile the embedding pipeline.

This script profiles the different stages of the embedding pipeline to identify
bottlenecks and measure the impact of optimizations.

Usage:
    python benchmark_pipeline.py [--n-songs N] [--song-duration D] [--gpus 0,1]

Example:
    python benchmark_pipeline.py --n-songs 20 --song-duration 30 --gpus 0,1
"""

import argparse
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import soxr

from audio_metrics.embedders.clap import LaionCLAP
from audio_metrics.embed import (
    resample,
    batch_accumulator,
    serialize_items,
    mix_pair,
    ItemCategory,
)
from audio_metrics.util.audio import multi_audio_slicer
from audio_metrics.util.cpu_parallel import cpu_parallel
from audio_metrics.util.gpu_parallel import gpu_parallel
from audio_metrics.mix_functions import mix_tracks_loudness
from audio_metrics.data import AudioMetricsData


@dataclass
class StageTimer:
    """Accumulates timing statistics for a pipeline stage."""

    name: str
    times: list = field(default_factory=list)

    def add(self, elapsed: float):
        self.times.append(elapsed)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0

    @property
    def count(self) -> int:
        return len(self.times)

    def __str__(self) -> str:
        if not self.times:
            return f"{self.name}: no data"
        return (
            f"{self.name:25s}: {self.total:8.2f}s total, "
            f"{self.mean * 1000:8.2f}ms avg, "
            f"{self.count:5d} items"
        )


def generate_synthetic_songs(n_songs, duration_sec, sr, stereo=True):
    """Generate synthetic audio data simulating songs.

    Args:
        n_songs: Number of songs to generate.
        duration_sec: Duration of each song in seconds.
        sr: Sample rate.
        stereo: If True, generate stereo (context+stem) pairs.

    Yields:
        Numpy arrays of shape (n_samples,) or (n_samples, 2).
    """
    n_samples = int(duration_sec * sr)
    for i in range(n_songs):
        if stereo:
            yield np.random.randn(n_samples, 2).astype(np.float32)
        else:
            yield np.random.randn(n_samples).astype(np.float32)


def benchmark_resampling(n_songs, duration_sec, sr_in, sr_out):
    """Benchmark resampling stage."""
    print(f"\n--- Resampling Benchmark ({n_songs} songs, {duration_sec}s each) ---")

    timer = StageTimer("resample")
    songs = list(generate_synthetic_songs(n_songs, duration_sec, sr_in, stereo=True))

    for song in songs:
        t0 = time.perf_counter()
        _ = soxr.resample(song, sr_in, sr_out)
        timer.add(time.perf_counter() - t0)

    print(f"  {timer}")
    print(f"  Throughput: {n_songs / timer.total:.2f} songs/sec")
    return timer


def benchmark_mixing(n_windows, win_dur, sr):
    """Benchmark mixing stage."""
    print(f"\n--- Mixing Benchmark ({n_windows} windows, {win_dur}s each) ---")

    timer = StageTimer("mix_tracks_loudness")
    n_samples = int(win_dur * sr)

    for _ in range(n_windows):
        audio = np.random.randn(n_samples, 2).astype(np.float32)
        t0 = time.perf_counter()
        _ = mix_tracks_loudness(audio, sr)
        timer.add(time.perf_counter() - t0)

    print(f"  {timer}")
    print(f"  Throughput: {n_windows / timer.total:.2f} windows/sec")
    return timer


def benchmark_gpu(model, n_batches, batch_size, device_indices):
    """Benchmark GPU embedding stage."""
    print(
        f"\n--- GPU Embedding Benchmark ({n_batches} batches, batch_size={batch_size}) ---"
    )

    # Generate batches
    n_samples = int(5.0 * 48000)  # 5s at 48kHz
    batches = [
        {
            "audio": np.random.randn(batch_size, n_samples).astype(np.float32),
            "category": np.zeros(batch_size, dtype=np.int32),
        }
        for _ in range(n_batches)
    ]

    # Warmup
    for batch in batches[:2]:
        _ = model.forward(batch)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    count = 0
    for result in gpu_parallel(
        iter(batches),
        model,
        target="forward",
        discard_input=False,
        device_indices=device_indices,
        in_buffer_size=len(device_indices) * 2,
        out_buffer_size=len(device_indices) * 2,
    ):
        count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {count / elapsed:.2f} batches/sec")
    print(f"  Throughput: {count * batch_size / elapsed:.2f} samples/sec")
    return elapsed, count


def benchmark_full_pipeline(
    n_songs, song_duration, sr_in, sr_out, win_dur, batch_size, device_indices
):
    """Benchmark the full embedding pipeline with synthetic data."""
    print(f"\n{'=' * 70}")
    print("Full Pipeline Benchmark")
    print(f"{'=' * 70}")
    print(f"  Songs: {n_songs}, Duration: {song_duration}s, SR: {sr_in}->{sr_out}")
    print(f"  Window: {win_dur}s, Batch: {batch_size}, GPUs: {device_indices}")

    # Generate synthetic songs
    songs = list(generate_synthetic_songs(n_songs, song_duration, sr_in, stereo=True))
    windows_per_song = int(song_duration / win_dur)
    total_windows = n_songs * windows_per_song
    total_batches = (total_windows + batch_size - 1) // batch_size

    print(f"  Total windows: {total_windows}, Total batches: ~{total_batches}")

    # Load model
    print("\nLoading CLAP model...")
    model = LaionCLAP()

    # Warmup
    print("Warming up GPU...")
    warmup_batch = {"audio": np.random.randn(8, int(win_dur * sr_out)).astype(np.float32)}
    for _ in range(3):
        _ = model.forward(warmup_batch)
    torch.cuda.synchronize()

    # Run pipeline stages with timing
    timers = {}

    # Stage 1: Resample
    print("\nStage 1: Resampling...")
    t0 = time.perf_counter()
    _resample = partial(resample, sr_orig=sr_in, sr_new=sr_out)
    resampled = list(
        cpu_parallel(
            iter(songs), _resample, n_workers=64, in_buffer_size=32, out_buffer_size=32
        )
    )
    timers["resample"] = time.perf_counter() - t0
    print(
        f"  Time: {timers['resample']:.2f}s ({n_songs / timers['resample']:.2f} songs/sec)"
    )

    # Stage 2: Slice into windows
    print("\nStage 2: Slicing...")
    t0 = time.perf_counter()
    windows = list(multi_audio_slicer(iter(resampled), win_dur, sr=sr_out))
    timers["slice"] = time.perf_counter() - t0
    print(f"  Time: {timers['slice']:.2f}s ({len(windows)} windows)")

    # Stage 3: Serialize (create aligned items for APA)
    print("\nStage 3: Serialize...")
    t0 = time.perf_counter()
    items = list(serialize_items(iter(windows), None, apa_mode=True, stems_mode=False))
    timers["serialize"] = time.perf_counter() - t0
    print(f"  Time: {timers['serialize']:.2f}s ({len(items)} items)")

    # Stage 4: Mix pairs
    print("\nStage 4: Mixing...")
    t0 = time.perf_counter()
    _mix_pair = partial(mix_pair, mix_func=mix_tracks_loudness, sr=sr_out)
    mixed = list(
        cpu_parallel(
            iter(items),
            _mix_pair,
            n_workers=64,
            discard_input=False,
            in_buffer_size=32,
            out_buffer_size=32,
        )
    )
    timers["mix"] = time.perf_counter() - t0
    print(f"  Time: {timers['mix']:.2f}s ({len(mixed)} items)")

    # Stage 5: Batch
    print("\nStage 5: Batching...")
    t0 = time.perf_counter()
    batches = list(batch_accumulator(iter(mixed), batch_size=batch_size))
    timers["batch"] = time.perf_counter() - t0
    print(f"  Time: {timers['batch']:.2f}s ({len(batches)} batches)")

    # Stage 6: GPU embedding
    print("\nStage 6: GPU Embedding...")
    t0 = time.perf_counter()
    results = list(
        gpu_parallel(
            iter(batches),
            model,
            target="forward",
            discard_input=False,
            device_indices=device_indices,
            in_buffer_size=len(device_indices) * 2,
            out_buffer_size=len(device_indices) * 2,
        )
    )
    torch.cuda.synchronize()
    timers["gpu"] = time.perf_counter() - t0
    print(f"  Time: {timers['gpu']:.2f}s ({len(results)} batches)")

    # Stage 7: Aggregation (simulated)
    print("\nStage 7: Aggregation...")
    t0 = time.perf_counter()
    metrics_data = AudioMetricsData(store_embeddings=False)
    for item in results:
        embedding = item["embedding"].cpu()
        metrics_data.add(embedding)
    timers["aggregate"] = time.perf_counter() - t0
    print(f"  Time: {timers['aggregate']:.2f}s")

    # Summary
    total = sum(timers.values())
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"{'Stage':<20} {'Time (s)':>10} {'%':>8}")
    print(f"{'-' * 40}")
    for stage, t in timers.items():
        print(f"{stage:<20} {t:>10.2f} {100 * t / total:>7.1f}%")
    print(f"{'-' * 40}")
    print(f"{'TOTAL':<20} {total:>10.2f} {100.0:>7.1f}%")
    print(f"\nOverall throughput: {n_songs / total:.2f} songs/sec")
    print(f"GPU utilization estimate: {100 * timers['gpu'] / total:.1f}%")

    return timers


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding pipeline")
    parser.add_argument("--n-songs", type=int, default=20, help="Number of songs")
    parser.add_argument(
        "--song-duration", type=float, default=30.0, help="Song duration (seconds)"
    )
    parser.add_argument("--sr-in", type=int, default=44100, help="Input sample rate")
    parser.add_argument(
        "--win-dur", type=float, default=5.0, help="Window duration (seconds)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--gpus", type=str, default=None, help="GPU indices (e.g., '0,1')"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick individual benchmarks only"
    )
    args = parser.parse_args()

    if args.gpus:
        device_indices = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        device_indices = list(range(torch.cuda.device_count()))

    sr_out = 48000  # CLAP sample rate

    print("=" * 70)
    print("Audio Metrics Pipeline Benchmark")
    print("=" * 70)
    print(f"GPUs: {device_indices}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.quick:
        # Quick individual benchmarks
        benchmark_resampling(10, args.song_duration, args.sr_in, sr_out)

        n_windows = int(10 * args.song_duration / args.win_dur)
        benchmark_mixing(n_windows, args.win_dur, sr_out)

        print("\nLoading CLAP model for GPU benchmark...")
        model = LaionCLAP()
        benchmark_gpu(model, 20, args.batch_size, device_indices)
    else:
        # Full pipeline benchmark
        benchmark_full_pipeline(
            n_songs=args.n_songs,
            song_duration=args.song_duration,
            sr_in=args.sr_in,
            sr_out=sr_out,
            win_dur=args.win_dur,
            batch_size=args.batch_size,
            device_indices=device_indices,
        )


if __name__ == "__main__":
    main()
