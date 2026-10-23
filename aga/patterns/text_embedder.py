
from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger("aga.text_embedder")

_DEFAULT_MODEL = "/Users/wuchunqi/.cache/modelscope/hub/models/sentence-transformers/msmarco-MiniLM-L-12-v3"

_EMBEDDER: Optional["TextEmbedder"] = None


class TextEmbedder:

    def __init__(self, model_name: str = _DEFAULT_MODEL, target_dim: int = 768,
                 allow_download: bool = True, offline_load_timeout: float = 20.0):
        self.model_name = model_name
        self.target_dim = int(target_dim)
        self.allow_download = allow_download
        self.offline_load_timeout = float(offline_load_timeout)
        self._model = None
        self._native_dim: Optional[int] = None
        self._adapter_W: Optional[np.ndarray] = None
        self._adapter_b: Optional[np.ndarray] = None
        self.available: bool = False
        self._tried_load = False

    def _load(self):
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.warning("sentence-transformers not installed; text-gen falls back to pseudo-embeddings.")
            return

        loaded = self._try_load(SentenceTransformer, local_files_only=True)
        if loaded:
            return
        if not self.allow_download:
            log.warning(
                "Text encoder %s not cached and downloads disabled; "
                "text-gen falls back to pseudo-embeddings.", self.model_name
            )
            return
        loaded = self._load_with_timeout(SentenceTransformer)
        if not loaded:
            log.warning(
                "Could not load/download %s within %.0fs; text-gen falls back "
                "to pseudo-embeddings.", self.model_name, self.offline_load_timeout
            )

    def _try_load(self, SentenceTransformer, local_files_only: bool) -> bool:
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=local_files_only)
            self._native_dim = int(self._model.get_sentence_embedding_dimension())
            self.available = True
            log.info("Loaded text encoder %s (native dim %d).", self.model_name, self._native_dim)
            return True
        except Exception:
            self._model = None
            return False

    def _load_with_timeout(self, SentenceTransformer) -> bool:
        import threading

        result: dict = {}

        def _worker():
            try:
                result["model"] = SentenceTransformer(self.model_name)
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=self.offline_load_timeout)
        if t.is_alive() or "error" in result or "model" not in result:
            return False
        self._model = result["model"]
        self._native_dim = int(self._model.get_sentence_embedding_dimension())
        self.available = True
        log.info("Loaded text encoder %s (native dim %d).", self.model_name, self._native_dim)
        return True

    def fit_adapter(self, real_texts: List[str], real_feats: np.ndarray) -> None:
        self._load()
        if real_feats.shape[0] == 0 or not self.available:
            return
        enc = self.encode_native(real_texts)
        k, nd = enc.shape
        ones = np.ones((k, 1), dtype=np.float32)
        X = np.concatenate([enc, ones], axis=1)
        Y = real_feats.astype(np.float32)
        lam = 1e-2
        A, *_ = np.linalg.lstsq(X, Y, rcond=None)
        XtX = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
        XtY = X.T @ Y
        A_reg = np.linalg.solve(XtX, XtY)
        self._adapter_W = A_reg[:nd].T
        self._adapter_b = A_reg[nd]
        log.info("Fit text->feature adapter: %d -> %d over %d real samples.",
                 nd, self.target_dim, k)

    def encode_native(self, texts: List[str]) -> np.ndarray:
        self._load()
        if self.available and self._model is not None:
            emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32)
        return self._pseudo_embed(texts, dim=self.target_dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        self._load()
        if not self.available:
            return self._pseudo_embed(texts, dim=self.target_dim)
        native = self.encode_native(texts)
        if self._adapter_W is not None and self._native_dim is not None:
            adapted = native @ self._adapter_W.T + self._adapter_b
            return adapted.astype(np.float32)
        if native.shape[1] == self.target_dim:
            return native.astype(np.float32)
        return self._resize(native, self.target_dim)

    @staticmethod
    def _pseudo_embed(texts: List[str], dim: int) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256((t or "").encode("utf-8")).digest()
            vals = np.frombuffer((h * ((dim // len(h)) + 1))[: dim * 4], dtype=np.uint8).astype(np.float32)
            vals = vals[:dim]
            vals = (vals / 255.0 - 0.5) * 2.0
            norm = np.linalg.norm(vals)
            if norm > 0:
                vals = vals / norm * 15.0
            out[i] = vals
        return out

    @staticmethod
    def _resize(emb: np.ndarray, target_dim: int) -> np.ndarray:
        d = emb.shape[1]
        if d == target_dim:
            return emb
        if d > target_dim:
            return emb[:, :target_dim]
        pad = np.zeros((emb.shape[0], target_dim - d), dtype=np.float32)
        return np.concatenate([emb, pad], axis=1)


def get_embedder(
    model_name: str = _DEFAULT_MODEL,
    target_dim: int = 768,
    allow_download: bool = True,
) -> "TextEmbedder":
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedder(
            model_name=model_name, target_dim=target_dim, allow_download=allow_download
        )
    return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER
    _EMBEDDER = None