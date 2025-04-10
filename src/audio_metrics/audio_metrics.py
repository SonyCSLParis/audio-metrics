import torch
from pathlib import Path
from audio_metrics.embed import embedding_pipeline, ItemCategory
from audio_metrics.data import AudioMetricsData
from audio_metrics.metrics.fad import frechet_distance
from audio_metrics.metrics.kd import kernel_distance
from audio_metrics.metrics.prdc import prdc
from audio_metrics.metrics.apa import apa, apa_compute_d_x_xp
from audio_metrics.projection import IncrementalPCA
from audio_metrics.mix_functions import MIX_FUNCTIONS, DEFAULT_MIX_FUNCTION
from audio_metrics.embedders import EMBEDDERS, DEFAULT_EMBEDDER
from audio_metrics.util.gpu_parallel import GPUWorkerHandler


class AudioMetrics:
    # metrics that need access to the full embeddings (not just mu, sigma)
    _need_embeddings = set(("kd", "precision", "prdc"))
    # for serialization
    _amd = (
        "stem_reference",
        "mix_reference",
        "mix_anti_reference",
        "stem_reference_pca",
        "mix_reference_pca",
        "mix_anti_reference_pca",
    )

    def __init__(
        self,
        metrics=["apa", "fad"],
        n_pca=None,
        device_indices=None,
        embedder=None,
        mix_function=None,
        win_dur=5.0,
        input_sr=None,
    ):
        self.gpu_handler = self._get_gpu_handler(device_indices)
        self.metrics = metrics
        self.need_apa = "apa" in self.metrics
        self.win_dur = win_dur
        self.input_sr = input_sr
        if n_pca is None:
            self.stem_projection = None
            self.mix_projection = None
        else:
            self.stem_projection = IncrementalPCA(n_components=n_pca)
            self.mix_projection = IncrementalPCA(n_components=n_pca)

        if embedder is None or isinstance(embedder, str):
            self.embedder = self.get_embedder(embedder)
        else:
            self.embedder = embedder

        if mix_function is None or isinstance(mix_function, str):
            self.mix_function = self.get_mix_function(mix_function)
        else:
            self.mix_function = mix_function

        self.apa_d_x_xp = None

        if self.need_apa:
            self.mix_reference = AudioMetricsData(self.store_mix_embeddings)
            self.mix_anti_reference = AudioMetricsData(self.store_mix_embeddings)
        else:
            self.mix_reference = None
            self.mix_anti_reference = None

        if self.stems_mode:
            self.stem_reference = AudioMetricsData(self.store_stem_embeddings)
        else:
            self.stem_reference = None

        self.mix_reference_pca = None
        self.mix_anti_reference_pca = None
        self.stem_reference_pca = None

    def save_state(self, fp: str | Path):
        state = self.__getstate__().copy()
        del state["mix_function"]
        del state["embedder"]
        del state["gpu_handler"]
        for attr in self._amd:
            item = state.get(attr)
            if item:
                state[attr] = item.serialize()
        for attr in ("stem_projection", "mix_projection"):
            item = state.get(attr)
            if item:
                state[attr] = item.__getstate__().copy()
        torch.save(state, fp)

    def load_state(self, fp: str | Path):
        state = torch.load(fp, weights_only=True)
        for attr in self._amd:
            item = state.get(attr)
            if item:
                state[attr] = AudioMetricsData.deserialize(item)
        for attr in ("stem_projection", "mix_projection"):
            item = state.get(attr)
            if item:
                getattr(self, attr).__setstate__(item)
                del state[attr]
        self.__dict__.update(state)

    @property
    def stems_mode(self):
        return any(metric for metric in self.metrics if metric != "apa")

    @property
    def store_mix_embeddings(self):
        return self.need_apa and self.mix_projection is not None

    @property
    def store_stem_embeddings(self):
        return self.stem_projection is not None or any(
            metric in self._need_embeddings for metric in self.metrics
        )

    def add_reference(self, reference):
        metrics = embedding_pipeline(
            reference,
            embedder=self.embedder,
            mix_function=self.mix_function,
            gpu_handler=self.gpu_handler,
            apa_mode="reference" if self.need_apa else None,
            stems_mode=self.stems_mode,
            store_mix_embeddings=self.store_mix_embeddings,
            store_stem_embeddings=self.store_stem_embeddings,
            win_dur=self.win_dur,
            input_sr=self.input_sr,
        )

        stem_reference = metrics.get(ItemCategory.stem)
        if stem_reference is not None:
            # invalidate cache:
            self.stem_reference_pca = None
            self.stem_reference += stem_reference
            self.stem_reference.recompute_stats()
        mix_reference = metrics.get(ItemCategory.aligned)
        if mix_reference is not None:
            # invalidate cache:
            self.mix_reference_pca = None
            self.mix_anti_reference_pca = None
            self.mix_reference += mix_reference

        mix_anti_reference = metrics.get(ItemCategory.misaligned)
        if mix_anti_reference is not None:
            self.mix_anti_reference += mix_anti_reference

    def reset_reference(self):
        if self.need_apa:
            self.apa_d_x_xp = None
            self.mix_reference = AudioMetricsData(self.store_mix_embeddings)
            self.mix_anti_reference = AudioMetricsData(self.store_mix_embeddings)
            self.mix_reference_pca = None
            self.mix_anti_reference_pca = None

        if self.stems_mode:
            self.stem_reference = AudioMetricsData(self.store_stem_embeddings)
            self.stem_reference_pca = None

    def ensure_stem_projection(self, ref, cand):
        if self.stem_projection is None:
            return ref, cand

        store_embs = any(metric in self._need_embeddings for metric in self.metrics)

        if self.stem_reference_pca is None:
            self.stem_projection.partial_fit(ref.embeddings)
            ref_emb = self.stem_projection.transform(ref.embeddings)
            ref = AudioMetricsData(store_embs)
            ref.add(ref_emb)
            self.stem_reference_pca = ref

        ref = self.stem_reference_pca

        cand_emb = self.stem_projection.transform(cand.embeddings)
        cand = AudioMetricsData(store_embs)
        cand.add(cand_emb)

        return ref, cand

    def ensure_mix_projection(self, ref, anti_ref, cand):
        if self.mix_projection is None:
            return ref, anti_ref, cand

        if self.mix_reference_pca is None:
            self.mix_projection.partial_fit(ref.embeddings)

            ref_emb = self.mix_projection.transform(ref.embeddings)
            anti_ref_emb = self.mix_projection.transform(anti_ref.embeddings)

            # only apa + fad for now, so no need for embeddings
            ref = AudioMetricsData(store_embeddings=False)
            anti_ref = AudioMetricsData(store_embeddings=False)

            ref.add(ref_emb)
            anti_ref.add(anti_ref_emb)

            self.mix_reference_pca = ref
            self.mix_anti_reference_pca = anti_ref

        ref, anti_ref = self.mix_reference_pca, self.mix_anti_reference_pca
        cand_emb = self.mix_projection.transform(cand.embeddings)
        cand = AudioMetricsData(store_embeddings=False)
        cand.add(cand_emb)

        return ref, anti_ref, cand

    def __call__(self, candidate):
        return self.evaluate(candidate)

    def evaluate(self, candidate):
        self.assert_reference()

        metrics = embedding_pipeline(
            candidate,
            embedder=self.embedder,
            mix_function=self.mix_function,
            gpu_handler=self.gpu_handler,
            apa_mode="candidate" if self.need_apa else None,
            stems_mode=self.stems_mode,
            store_mix_embeddings=self.store_mix_embeddings,
            store_stem_embeddings=self.store_stem_embeddings,
            win_dur=self.win_dur,
            input_sr=self.input_sr,
        )

        stem_cand = metrics.get(ItemCategory.stem)
        apa_cand = metrics.get(ItemCategory.aligned)
        stem_ref = self.stem_reference
        apa_ref = self.mix_reference
        apa_anti_ref = self.mix_anti_reference

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

        if "prdc" in self.metrics:
            k = max(1, min(10, len(stem_ref), len(stem_cand)))
            result.update(prdc(stem_ref, stem_cand, k))

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

    def get_mix_function(self, mix_function):
        if mix_function is None:
            mix_function = DEFAULT_MIX_FUNCTION
        func = MIX_FUNCTIONS.get(mix_function)
        if func is None:
            msg = f"Unknown mix_function {mix_function}, must be one of {MIX_FUNCTIONS.keys()}"
            raise ValueError(msg)
        return func

    def get_embedder(self, embedder):
        if embedder is None:
            embedder = DEFAULT_EMBEDDER
        info = EMBEDDERS.get(embedder)
        if info is None:
            msg = f"Unknown embedder {embedder}, must be one of {EMBEDDERS.keys()}"
            raise ValueError(msg)
        cls, kwargs = info
        return cls(**kwargs)

    def assert_reference(self):
        msg = (
            "The reference dataset is empty. This can have various causes:"
            "  - You have not called AudioMetrics.add_reference()"
            "  - You have called AudioMetrics.add_reference() with an empty dataset"
            "  - The duration of your audio is shorter than `win_dur` ({self.win_dur}s)."
            "    (You can specify your own `win_dur` when instantiating AudioMetrics)"
        )
        if self.stems_mode:
            if self.stem_reference.n is None:
                raise ValueError(msg)
        if self.need_apa:
            if self.mix_reference.n is None:
                raise ValueError(msg)
