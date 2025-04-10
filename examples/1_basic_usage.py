import numpy as np
from audio_metrics import AudioMetrics

sr = 48000
n_seconds = 5

n_windows = 100
window_len = sr * n_seconds

reference = np.random.random((n_windows, window_len))
candidate = np.random.random((n_windows, window_len))

metrics = AudioMetrics(
    metrics=[
        "prdc",  # precision, recall, density, coverage
        "fad",  # frechet audio distance
        "kd",  # kernel distance
    ],
    input_sr=sr,
)
metrics.add_reference(reference)

print(metrics.evaluate(candidate))

# To compute APA, the input data must be pairs of context and stem (in the
# trailing dimension)
reference = np.random.random((n_windows, window_len, 2))
# Data can also be passed as a generator, to facilitate processing larger
# datasets
candidate = (np.random.random((window_len, 2)) for _ in range(n_windows))

# stem-only metrics (like FAD), can be computed simultaneously with APA
metrics = AudioMetrics(
    metrics=["fad", "apa"],
    input_sr=sr,
)
metrics.add_reference(reference)
print(metrics.evaluate(candidate))
