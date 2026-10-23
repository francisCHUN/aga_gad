
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import dgl

from ..data import loader
from ..models.split import make_split
from ..orchestration.schema import (
    DatasetReport,
    GapReview,
    GenerationStrategy,
    MissingPatterns,
)

log = logging.getLogger("aga.cache")

CACHE_VERSION = 1
_GRAPH_FILE = "augmented_graph.bin"
_META_FILE = "meta.json"
_SECRET_FIELDS = {"tokens", "api_key", "key", "secret", "authorization"}

_impl_fp: Optional[str] = None

DEFAULT_GENERATION_SCOPE = "gen-v1"


def impl_fingerprint(scope: Optional[str] = None) -> str:
    if scope in (None, "", "default"):
        return DEFAULT_GENERATION_SCOPE
    if scope != "code":
        return "tag-" + scope
    global _impl_fp
    if _impl_fp is None:
        root = Path(__file__).resolve().parent.parent
        h = hashlib.md5()
        files = sorted(root.rglob("*.py"))
        if not files:
            _impl_fp = "unknown"
            return _impl_fp
        for f in files:
            try:
                h.update(str(f.relative_to(root)).encode("utf-8"))
                h.update(b"\0")
                h.update(hashlib.md5(f.read_bytes()).digest())
            except OSError:
                _impl_fp = "unreadable"
                return _impl_fp
        _impl_fp = h.hexdigest()[:12]
    return _impl_fp


def data_fingerprint(data_dir: str, dataset: str) -> str:
    key = dataset.strip()
    spec = loader.DATASETS.get(key) or loader.DATASETS.get(key.capitalize()) \
        or {"mat": f"{key}.mat", "csv": f"{key}.csv"}
    h = hashlib.md5()
    for rel in (spec["mat"], spec["csv"]):
        p = os.path.join(data_dir, rel)
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{rel}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
        else:
            h.update(f"{rel}|missing".encode("utf-8"))
    return h.hexdigest()[:12]


def build_cache_key(
    *,
    dataset: str,
    data_dir: str,
    backend: str,
    llm_cfg: Optional[Dict],
    split_cfg: Optional[Dict],
    generation_seed: int,
    text_gen: bool,
    embed_model: str,
    allow_download: bool,
    generation_scope: Optional[str] = None,
    review: Optional[Dict] = None,
) -> Dict[str, Any]:
    import os as _os

    if generation_scope is None:
        generation_scope = _os.environ.get("CACHE_SCOPE")
    sc = split_cfg or {}
    filtered_llm = {
        k: v for k, v in (llm_cfg or {}).items()
        if k not in _SECRET_FIELDS
    }
    return {
        "cache_version": CACHE_VERSION,
        "dataset": dataset,
        "data_fingerprint": data_fingerprint(data_dir, dataset),
        "impl_fingerprint": impl_fingerprint(generation_scope),
        "backend": backend,
        "llm_cfg": filtered_llm,
        "split": {
            "seed": int(sc.get("seed", 42)),
            "val_ratio": float(sc.get("val_ratio", 0.10)),
            "test_ratio": float(sc.get("test_ratio", 0.20)),
            "budget_ratio": float(sc.get("budget_ratio", 0.1)),
        },
        "generation_seed": int(generation_seed),
        "text_gen": bool(text_gen),
        "embed_model": str(embed_model),
        "allow_download": bool(allow_download),
        "review": dict(review or {}),
    }


def key_digest(key: Dict[str, Any]) -> str:
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:8]


def entry_dir(cache_dir: str, key: Dict[str, Any]) -> Path:
    return Path(cache_dir) / f"{key['dataset']}-{key_digest(key)}"


def gap_loop_cache_component(
    review_config: Optional[Dict],
    sampling_cfg: "ReviewSamplingConfig",
    backend: str,
    llm_cfg: Optional[Dict],
) -> Dict[str, Any]:
    filtered_llm = {
        k: v for k, v in (llm_cfg or {}).items()
        if k not in _SECRET_FIELDS
    }
    rc = dict(review_config or {})
    for k in ("llm_precheck", "coverage_criterion"):
        rc.setdefault(k, None)
    rc.pop("targeted", None); rc.pop("stratified", None); rc.pop("boundary", None)
    return {
        "mode": "gap_loop",
        "loop_config": {k: (str(v) if v is not None else None)
                        for k, v in sorted(rc.items())},
        "sampling": {
            "targeted": int(sampling_cfg.targeted),
            "stratified": int(sampling_cfg.stratified),
            "boundary": int(sampling_cfg.boundary),
            "seed": int(sampling_cfg.seed),
        },
        "backend": backend,
        "llm_cfg": filtered_llm,
    }


def save_augmented(cache_dir: str, key: Dict[str, Any], result) -> Path:
    d = entry_dir(cache_dir, key)
    d.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(d / _GRAPH_FILE), [result.augmented_graph])
    meta = {
        "key": key,
        "key_digest": key_digest(key),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "num_synthetic": int(result.num_synthetic),
        "text_gen": bool(result.text_gen),
        "split_counts": {
            "n_train": int(result.split.get("n_train", -1)),
            "n_val": int(result.split.get("n_val", -1)),
            "n_test": int(result.split.get("n_test", -1)),
        },
        "report": result.report.model_dump_json(),
        "missing": result.missing.model_dump_json(),
        "strategy": result.strategy.model_dump_json(),
        "reviewed_gaps": [
            r.model_dump_json() for r in getattr(result, "reviewed_gaps", [])
        ],
    }
    with open(d / _META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return d


def load_augmented(
    cache_dir: str,
    key: Dict[str, Any],
    *,
    data_dir: str,
    dataset: str,
) -> Optional[Dict[str, Any]]:
    d = entry_dir(cache_dir, key)
    meta_path, graph_path = d / _META_FILE, d / _GRAPH_FILE
    if not (meta_path.is_file() and graph_path.is_file()):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("key") != key:
            log.warning("Cache entry %s has a stale key; regenerating.", d)
            return None

        g, _ = loader.load_by_name(dataset, data_dir)
        split = make_split(
            g,
            seed=key["split"]["seed"],
            val_ratio=key["split"]["val_ratio"],
            test_ratio=key["split"]["test_ratio"],
        )

        graphs, _ = dgl.load_graphs(str(graph_path))
        augmented_graph = graphs[0]

        return {
            "report": DatasetReport.model_validate_json(meta["report"]),
            "missing": MissingPatterns.model_validate_json(meta["missing"]),
            "strategy": GenerationStrategy.model_validate_json(meta["strategy"]),
            "baseline_graph": g,
            "augmented_graph": augmented_graph,
            "split": split,
            "num_synthetic": int(meta["num_synthetic"]),
            "text_gen": bool(meta["text_gen"]),
            "reviewed_gaps": [
                GapReview.model_validate_json(blob)
                for blob in meta.get("reviewed_gaps", [])
            ],
        }
    except Exception:
        log.warning("Cache entry %s is unreadable; regenerating.", d, exc_info=True)
        return None