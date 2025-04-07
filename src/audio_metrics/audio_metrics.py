import torch
from pathlib import Path
from audio_metrics.embed import embedding_pipeline, ItemCategory
from audio_metrics.data import AudioMetricsData
from audio_metrics.metrics.fad import frechet_distance
from audio_metrics.metrics.kid import kernel_distance
from audio_metrics.metrics.apa import apa, apa_compute_d_x_xp
from audio_metrics.projection import IncrementalPCA
from audio_metrics.embedders.clap import CLAP
from audio_metrics.util.gpu_parallel import GPUWorkerHandler


class AudioMetrics:
    # metrics that need access to the full embeddings (not just mu, sigma)
    _need_embeddings = set(("kd", "precision", "recall", "coverage", "density"))

    def __init__(
        self, metrics=["apa", "fad"], n_pca=None, device_indices=None, embedder=None
    ):
        self.gpu_handler = self._get_gpu_handler(device_indices)
        self.metrics = metrics
        self.need_apa = "apa" in self.metrics

        if n_pca is None:
            self.stem_projection = None
            self.mix_projection = None
        else:
            self.stem_projection = IncrementalPCA(n_components=n_pca)
            self.mix_projection = IncrementalPCA(n_components=n_pca)

        if embedder is None:
            self.embedder = self.get_embedder()
        else:
            self.embedder = embedder

        self.apa_d_x_xp = None

        if self.need_apa:
            self.apa_reference = AudioMetricsData(self.store_mix_embeddings)
            self.apa_anti_reference = AudioMetricsData(self.store_mix_embeddings)
        else:
            self.apa_reference = None
            self.apa_anti_reference = None

        if self.stems_mode:
            self.stem_reference = AudioMetricsData(self.store_stem_embeddings)
        else:
            self.stem_reference = None

        self._cached_mix_pca_projections = None
        self._cached_stem_pca_projections = None

    def save_state(self, fp: str | Path):
        state = self.__getstate__()
        del state["embedder"]
        del state["gpu_handler"]
        torch.save(state, fp)

    def load_state(self, fp: str | Path):
        state = torch.load(fp, weights_only=True)
        self.__setstate__(state)

    def _load_metrics_data(self, state):
        attrs = ["apa_reference", "apa_anti_reference", "stem_reference"]
        for attr in attrs:
            if attr in state:
                setattr(self, attr, AudioMetricsData())
                getattr(self, attr).__setstate__(state[attr])

    @property
    def stems_mode(self):
        return any(metric for metric in self.metrics if metric != "apa")

    @property
    def store_mix_embeddings(self):
        return self.need_apa and self.mix_projection

    @property
    def store_stem_embeddings(self):
        return self.stem_projection or any(
            metric in self._need_embeddings for metric in self.metrics
        )

    def add_reference(self, reference):
        metrics = embedding_pipeline(
            reference,
            embedder=self.embedder,
            gpu_handler=self.gpu_handler,
            apa_mode="reference" if self.need_apa else None,
            stems_mode=self.stems_mode,
            store_mix_embeddings=self.store_mix_embeddings,
            store_stem_embeddings=self.store_stem_embeddings,
        )

        stem_reference = metrics.get(ItemCategory.stem)
        if stem_reference is not None:
            # invalidate cache:
            self._cached_stem_pca_projections = None
            self.stem_reference += stem_reference

        apa_reference = metrics.get(ItemCategory.aligned)
        if apa_reference is not None:
            # invalidate cache:
            self._cached_apa_pca_projections = None
            self.apa_reference += apa_reference

        apa_anti_reference = metrics.get(ItemCategory.misaligned)
        if apa_anti_reference is not None:
            self.apa_anti_reference += apa_anti_reference

    def reset_reference(self):
        if self.need_apa:
            self.apa_d_x_xp = None
            self.apa_reference = AudioMetricsData(self.store_mix_embeddings)
            self.apa_anti_reference = AudioMetricsData(self.store_mix_embeddings)
            self._cached_mix_pca_projections = None

        if self.stems_mode:
            self.stem_reference = AudioMetricsData(self.store_stem_embeddings)
            self._cached_stem_pca_projections = None

    def ensure_stem_projection(self, ref, cand):
        if self.stem_projection is None:
            return ref, cand

        store_embs = any(metric in self._need_embeddings for metric in self.metrics)

        if self._cached_stem_pca_projections is None:
            self.stem_projection.partial_fit(ref.embeddings)
            ref_emb = self.stem_projection.transform(ref.embeddings)
            ref = AudioMetrics(store_embs)
            ref.add(ref_emb)
            self._cached_stem_pca_projections = ref

        ref = self._cached_stem_pca_projections

        cand_emb = self.stem_projection.transform(cand.embeddings)
        cand = AudioMetrics(store_embs)
        cand.add(cand_emb)

        return ref, cand

    def ensure_mix_projection(self, ref, anti_ref, cand):
        if self.mix_projection is None:
            return ref, anti_ref, cand

        if self._cached_mix_pca_projections is None:
            self.mix_projection.partial_fit(ref.embeddings)

            ref_emb = self.mix_projection.transform(ref.embeddings)
            anti_ref_emb = self.mix_projection.transform(anti_ref.embeddings)

            # only apa + fad for now, so no need for embeddings
            ref = AudioMetricsData(store_embeddings=False)
            anti_ref = AudioMetricsData(store_embeddings=False)

            ref.add(ref_emb)
            anti_ref.add(anti_ref_emb)

            self._cached_stem_pca_projections = (ref, anti_ref)

        ref, anti_ref = self._cached_stem_pca_projections

        cand_emb = self.mix_projection.transform(cand.embeddings)
        cand = AudioMetricsData(store_embeddings=False)
        cand.add(cand_emb)

        return ref, anti_ref, cand

    def __call__(self, candidate):
        self.assert_reference()

        metrics = embedding_pipeline(
            candidate,
            embedder=self.embedder,
            gpu_handler=self.gpu_handler,
            apa_mode="candidate" if self.need_apa else None,
            stems_mode=self.stems_mode,
            store_mix_embeddings=self.store_mix_embeddings,
            store_stem_embeddings=self.store_stem_embeddings,
        )

        stem_cand = metrics.get(ItemCategory.stem)
        apa_cand = metrics.get(ItemCategory.aligned)
        stem_ref = self.stem_reference
        apa_ref = self.apa_reference
        apa_anti_ref = self.apa_anti_reference

        if self.stems_mode and stem_cand is None:
            raise ValueError("No stem candidate embeddings were computed")

        if self.need_apa and apa_cand is None:
            raise ValueError("No apa candidate embeddings were computed")

        if self.stems_mode:
            stem_ref, stem_cand = self.ensure_stem_projection(stem_ref, stem_cand)

        if self.need_apa:
            apa_ref, apa_anti_ref, apa_cand = self.ensure_mix_projection(
                apa_ref,
                apa_anti_ref,
                apa_cand,
            )
            if self.apa_d_x_xp is None:
                self.apa_d_x_xp = apa_compute_d_x_xp(apa_ref, apa_anti_ref)

        result = {}

        if "fad" in self.metrics:
            result["fad"] = frechet_distance(stem_cand, stem_ref)

        if "kd" in self.metrics:
            result.update(kernel_distance(stem_cand, stem_ref))

        # TODO: prec, recall, density, coverage

        if self.need_apa:
            result["apa"] = apa(
                apa_cand,
                apa_ref,
                apa_anti_ref,
                self.apa_d_x_xp,
            )

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
            # assert (
            #     self.stem_reference
            # ), f"To compute stem metrics, specify one or more of {self._need_stems}"
            assert (
                self.stem_reference.n
            ), "Call AudioMetrics.add_reference() at least once before evaluating candidates"
        if self.need_apa:
            assert (
                self.apa_reference.n
            ), "Call AudioMetrics.add_reference() at least once before evaluating candidates"
