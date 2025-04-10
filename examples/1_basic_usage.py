import numpy as np
from audio_metrics import AudioMetrics

sr = 48000
n_seconds = 5

n_windows = 100
window_len = sr * n_seconds

reference = np.random.random((n_windows, window_len))
candidate = np.random.random((n_windows, window_len))

metrics = AudioMetrics(metrics=["fad", "prdc", "kd"])
metrics.add_reference(reference)

print(metrics.evaluate(candidate))

# To compute APA, the input data must be pairs of context and stem (in the
# trailing dimension)
reference = np.random.random((n_windows, window_len, 2))
# Data can also be passed as a generator, to facilitate processing larger
# datasets
candidate = (np.random.random((window_len, 2)) for _ in range(n_windows))

metrics = AudioMetrics(metrics=["fad", "prdc", "apa"])
metrics.add_reference(reference)
print(metrics.evaluate(candidate))
