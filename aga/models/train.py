
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import inspect

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

try:
    import dgl
except ImportError as exc:
    raise ImportError("DGL is required; use the python312 env.") from exc

from .registry import build_model


@dataclass
class TrainConfig:
    model: str = "dominant"
    epochs: int = 100
    lr: float = 1e-3
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.2
    device: str = "cpu"
    attr_weight: float = 1.0
    stru_weight: float = 1.0
    seed: int = 42
    class_weight: str = "inverse"
    model_kwargs: Dict = field(default_factory=dict)


def compute_class_weight(
    anomaly: torch.Tensor,
    train_mask: torch.Tensor,
    mode: str = "inverse",
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    y = anomaly[train_mask].long()
    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    ratio = n_neg / n_pos
    w_pos = ratio ** 0.5 if mode == "sqrt" else float(ratio)
    dev = device if device is not None else anomaly.device
    return torch.tensor([1.0, float(w_pos)], dtype=torch.float32, device=dev)


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_evaluate(
    g: "dgl.DGLGraph",
    cfg: TrainConfig,
    eval_mask_name: str = "test_mask",
    verbose: bool = False,
) -> Dict[str, float]:
    _set_seed(cfg.seed)

    device = torch.device("cpu")

    g = g.to(device)
    x = g.ndata["feat"].float()
    anomaly = g.ndata["anomaly"].long()
    train_mask = g.ndata["train_mask"].bool()

    build_kwargs = dict(
        in_dim=x.shape[1],
        hidden=cfg.hidden,
        n_layers=cfg.layers,
        dropout=cfg.dropout,
        attr_weight=cfg.attr_weight,
        stru_weight=cfg.stru_weight,
    )
    build_kwargs.update(cfg.model_kwargs or {})
    model = build_model(cfg.model, **build_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    loss_kwargs: Dict = {}
    cw = compute_class_weight(anomaly, train_mask, cfg.class_weight, device)
    if cw is not None and "class_weight" in inspect.signature(model.loss).parameters:
        loss_kwargs["class_weight"] = cw

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = model.loss(g, x, mask=train_mask, anomaly=anomaly,
                          train_mask=train_mask, **loss_kwargs)
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  [epoch {epoch+1:3d}] loss={loss.item():.4f}")

    model.eval()
    scores = model.score(g, x, attr_weight=cfg.attr_weight, stru_weight=cfg.stru_weight).cpu().numpy() \
        if cfg.model == "dominant" else model.score(g, x).cpu().numpy()
    labels = anomaly.cpu().numpy()
    eval_mask = g.ndata[eval_mask_name].bool().cpu().numpy()

    y_true = labels[eval_mask]
    y_score = scores[eval_mask]
    has_two_classes = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_score)) if has_two_classes else float("nan")
    ap = float(average_precision_score(y_true, y_score)) if has_two_classes else float("nan")
    y_pred = (y_score > np.median(y_score)).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {"auc": auc, "ap": ap, "f1": f1, "num_eval": int(eval_mask.sum())}


def _setup_cjk_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _cjk_candidates = [
        "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
        "Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei",
        "Source Han Sans SC", "WenQuanYi Zen Hei",
    ]
    _available = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["axes.unicode_minus"] = False
    for _cjk in _cjk_candidates:
        if _cjk in _available:
            plt.rcParams["font.sans-serif"] = [_cjk] + plt.rcParams["font.sans-serif"]
            break


def _ensure_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return True
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting; install with `pip install matplotlib`."
        ) from exc


def visualize_tsne_compare(
    baseline_g: "dgl.DGLGraph",
    augmented_g: "dgl.DGLGraph",
    out_path: str = "tsne_compare.png",
    seed: int = 42,
) -> str:
    _ensure_matplotlib()
    _setup_cjk_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    feat_aug = augmented_g.ndata["feat"].float().cpu().numpy()
    anomaly = augmented_g.ndata["anomaly"].long().cpu().numpy()
    is_synth = augmented_g.ndata["is_synthetic"].bool().cpu().numpy()
    n_base = int(baseline_g.num_nodes())

    perplexity = min(30, max(5, feat_aug.shape[0] // 8))
    embed = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(feat_aug)

    original_normal = (~is_synth) & (anomaly == 0)
    original_anomaly = (~is_synth) & (anomaly == 1)
    synthetic_anomaly = is_synth

    style = {
        "Normal":    (original_normal,    "#C3DAEA", "o", 12, 1, "none", 0.0),
        "Anomaly":   (original_anomaly,   "#F5A8AA", "o", 24, 1, "none", 0.0),
        "Generated": (synthetic_anomaly,  "#787AB9", "o", 30, 1, "black", 0.8),
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for label, (mask, color, marker, size, alpha, edgecolor, linewidth) in style.items():
        if int(mask.sum()) == 0:
            continue
        ax.scatter(
            embed[mask, 0], embed[mask, 1],
            c=color, marker=marker, s=size, alpha=alpha,
            label=f"{label} ({int(mask.sum())})",
            edgecolors=edgecolor, linewidths=linewidth,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def train_compare(
    baseline_g: "dgl.DGLGraph",
    augmented_g: "dgl.DGLGraph",
    cfg: TrainConfig,
    eval_mask_name: str = "test_mask",
    visualize: bool = False,
    plot_path: str = "tsne_compare.png",
) -> Dict[str, Dict[str, float]]:
    b_mask = baseline_g.ndata[eval_mask_name].bool()
    a_mask = augmented_g.ndata[eval_mask_name].bool()
    b_ids = set(torch.where(b_mask)[0].tolist())
    a_ids = set(torch.where(a_mask)[0].tolist())
    assert b_ids == a_ids, (
        f"Fairness contract broken: {eval_mask_name} node sets differ between "
        f"baseline and augmented graphs."
    )

    baseline = train_and_evaluate(baseline_g, cfg, eval_mask_name=eval_mask_name, verbose=False)
    augmented = train_and_evaluate(augmented_g, cfg, eval_mask_name=eval_mask_name, verbose=False)

    if visualize:
        visualize_tsne_compare(baseline_g, augmented_g, out_path=plot_path, seed=cfg.seed)

    return {"baseline": baseline, "augmented": augmented}


def _nearest_cosine(H: np.ndarray, query_idx: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    if len(ref_idx) == 0 or len(query_idx) == 0:
        return np.asarray([], dtype=np.float32)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    sims = Hn[query_idx] @ Hn[ref_idx].T
    return sims.max(axis=1).astype(np.float32)


def embedding_similarity_check(
    baseline_g: "dgl.DGLGraph",
    augmented_g: "dgl.DGLGraph",
    out_path: str = "embedding_similarity.png",
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    _set_seed(seed)

    anomaly = augmented_g.ndata["anomaly"].long().cpu().numpy()
    is_synth = augmented_g.ndata["is_synthetic"].bool().cpu().numpy()
    orig_anomaly_idx = np.where((~is_synth) & (anomaly == 1))[0]
    synthetic_idx = np.where(is_synth)[0]

    n_orig = int(baseline_g.num_nodes())
    assert n_orig == augmented_g.num_nodes() - len(synthetic_idx), (
        f"Original node count mismatch: baseline has {n_orig} nodes but augmented "
        f"graph has {len(synthetic_idx)} synthetic nodes out of "
        f"{augmented_g.num_nodes()} total."
    )

    H = augmented_g.ndata["feat"].float().cpu().numpy()
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)

    if len(synthetic_idx) and len(orig_anomaly_idx):
        mat_anomaly = Hn[synthetic_idx] @ Hn[orig_anomaly_idx].T
    else:
        mat_anomaly = np.empty((len(synthetic_idx), 0), dtype=np.float32)
    sim_to_anomaly = mat_anomaly.max(axis=1).astype(np.float32) if mat_anomaly.size else \
        np.asarray([], dtype=np.float32)

    if len(synthetic_idx) > 1:
        mat_synthetic = Hn[synthetic_idx] @ Hn[synthetic_idx].T
        np.fill_diagonal(mat_synthetic, np.nan)
        sim_to_synthetic = np.nanmax(mat_synthetic, axis=1).astype(np.float32)
    elif len(synthetic_idx) == 1:
        mat_synthetic = np.full((1, 1), np.nan, dtype=np.float32)
        sim_to_synthetic = np.asarray([], dtype=np.float32)
    else:
        mat_synthetic = np.empty((0, 0), dtype=np.float32)
        sim_to_synthetic = np.asarray([], dtype=np.float32)

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    stats = {
        "to_anomaly": _stats(sim_to_anomaly),
        "to_synthetic": _stats(sim_to_synthetic),
        "n_synthetic": int(len(synthetic_idx)),
        "n_orig_anomaly": int(len(orig_anomaly_idx)),
    }

    if len(synthetic_idx) == 0:
        print(f"[embedding_similarity_check] no synthetic nodes; skipping plot "
              f"(n_synthetic=0).")
        return stats

    _ensure_matplotlib()
    _setup_cjk_matplotlib()
    import matplotlib.pyplot as plt

    series = [
        ("to_anomaly",   sim_to_anomaly,   "合成异常 ↔ 最近原始异常", "#DD8452"),
        ("to_synthetic", sim_to_synthetic, "合成异常 ↔ 最近合成异常", "#C44E52"),
    ]

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax_hist = fig.add_subplot(gs[0, 0])

    bins = np.linspace(0.0, 1.0, 31)
    for _key, arr, label, color in series:
        if arr.size == 0:
            continue
        ax_hist.hist(np.clip(arr, 0.0, 1.0), bins=bins, alpha=0.55, color=color,
                     edgecolor="white", label=f"{label} (n={arr.size})")
    ax_hist.set_xlabel("cosine similarity (最近邻)")
    ax_hist.set_ylabel("合成异常节点数")
    ax_hist.set_title("最近邻余弦相似度分布 (原始特征, 送入模型前)")
    ax_hist.legend(loc="best", fontsize=9)
    ax_hist.set_xlim(0.0, 1.0)

    notes = []
    if sim_to_anomaly.size == 0:
        notes.append("原始异常样本: N/A")
    if sim_to_synthetic.size == 0:
        notes.append("合成-合成: N/A")
    if notes:
        ax_hist.text(0.98, 0.98, "\n".join(notes), ha="right", va="top",
                     transform=ax_hist.transAxes, fontsize=9, color="#666666")

    def _draw_heatmap(ax, M, title, xlab):
        if M is None or M.size == 0 or (M.shape[0] == 0 or M.shape[1] == 0):
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#666666")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            return
        im = ax.imshow(np.clip(M, 0.0, 1.0), aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("合成异常节点", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    _draw_heatmap(
        fig.add_subplot(gs[0, 1]),
        mat_anomaly,
        "热力图 合成异常 ↔ 原始异常 (cosine)",
        f"原始异常节点 (n={len(orig_anomaly_idx)})",
    )

    fig.suptitle(
        "合成异常 vs 最近原始异常 / 最近合成异常 — 原始 Embedding 余弦相似度",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats