from pathlib import Path
from audio_metrics.embed import embedding_pipeline, ItemCategory
from audio_metrics.data import AudioMetricsData
from audio_metrics.metrics.fad import frechet_distance
from audio_metrics.metrics.kid import kernel_distance
from audio_metrics.metrics.apa import apa, apa_compute_d_x_xp

from audio_metrics.embedders.clap import CLAP
from audio_metrics.util.gpu_parallel import GPUWorkerHandler


class AudioMetrics:
    need_embeddings = set(("kd", "precision", "recall"))
    need_stems = set(("fad", "kd", "precision", "recall"))

    def __init__(self, metrics=["apa", "fad"], device_indices=None, embedder=None):
        self.gpu_handler = self._get_gpu_handler(device_indices)
        if embedder is None:
            self.embedder = self.get_embedder()
        else:
            self.embedder = embedder
        self.metrics = metrics
        self.need_apa = "apa" in self.metrics
        self.store_embeddings = self.need_embeddings.intersection(self.metrics)
        self.stems_mode = self.need_stems.intersection(self.metrics)
        self.stem_reference = None
        self.apa_reference = None
        self.apa_anti_reference = None
        self.apa_d_x_xp = None
        if self.need_apa:
            self.apa_reference = AudioMetricsData()
            self.apa_anti_reference = AudioMetricsData()
        if self.stems_mode:
            self.stem_reference = AudioMetricsData(self.store_embeddings)

    def add_reference(self, reference):
        apa_mode = "reference" if self.need_apa else None
        if self.stems_mode:
            stems_mode = "embeddings" if self.store_embeddings else "stats"
        else:
            stems_mode = None
        metrics = embedding_pipeline(
            reference,
            embedder=self.embedder,
            gpu_handler=self.gpu_handler,
            apa_mode=apa_mode,
            stems_mode=stems_mode,
        )
        stem_reference = metrics.get(ItemCategory.stem)
        if stem_reference is not None:
            self.stem_reference += stem_reference
        apa_reference = metrics.get(ItemCategory.aligned)
        if apa_reference is not None:
            self.apa_reference += apa_reference
        apa_anti_reference = metrics.get(ItemCategory.misaligned)
        if apa_anti_reference is not None:
            self.apa_anti_reference += apa_anti_reference
        if apa_mode:
            self.apa_d_x_xp = apa_compute_d_x_xp(
                self.apa_reference, self.apa_anti_reference
            )

    def reset_reference(self):
        self.apa_d_x_xp = None
        self.apa_reference = AudioMetricsData()
        self.apa_anti_reference = AudioMetricsData()

    def __call__(self, candidate):
        self.assert_reference()
        apa_mode = "candidate" if self.need_apa else None
        if self.stems_mode:
            stems_mode = "embeddings" if self.store_embeddings else "stats"
        else:
            stems_mode = None
        metrics = embedding_pipeline(
            candidate,
            embedder=self.embedder,
            gpu_handler=self.gpu_handler,
            apa_mode=apa_mode,
            stems_mode=stems_mode,
        )
        stem_candidate = metrics.get(ItemCategory.stem)
        apa_candidate = metrics.get(ItemCategory.aligned)
        if self.stems_mode and stem_candidate is None:
            raise ValueError("No stem candidate embeddings were computed")

        if self.need_apa and apa_candidate is None:
            raise ValueError("No apa candidate embeddings were computed")
        result = {}
        if "fad" in self.metrics:
            result["fad"] = frechet_distance(
                stem_candidate,
                self.stem_reference,
            ).item()
        if "kd" in self.metrics:
            result.update(
                kernel_distance(
                    stem_candidate.embeddings,
                    self.stem_reference.embeddings,
                )
            )
        if self.need_apa:
            result["apa"] = apa(
                apa_candidate,
                self.apa_reference,
                self.apa_anti_reference,
                self.apa_d_x_xp,
            ).item()
        return result

    def _get_gpu_handler(self, device_indices):
        if device_indices or device_indices is None:
            return GPUWorkerHandler(device_indices)
        return None

    def get_embedder(self):
        clap_cktpt = (
            "/home/maarten/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt"
        )
        # return benedict({"sr": 48000})
        return CLAP(ckpt=clap_cktpt)

    def assert_reference(self):
        if self.stems_mode:
            assert self.stem_reference, (
                f"To compute stem metrics, specify one or more of {self.need_stems}"
            )
            assert self.stem_reference.n, (
                "Call AudioMetrics.add_reference() at least once before evaluating candidates"
            )
        if self.need_apa:
            assert self.apa_reference.n, (
                "Call AudioMetrics.add_reference() at least once before evaluating candidates"
            )
