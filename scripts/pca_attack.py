"""PCA visualization of heatmaps: original vs adversarial vs delta, per model.

Usage:
    python pca_attack.py            # without delta
    python pca_attack.py --delta    # with delta
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = "/net/tscratch/people/plgpkitlo/Audio-XAI/runs/attack"
MODELS = {
    "AST": "ast_epoch=4-step=9740_test_100_samples",
    "VGGish": "vggish_epoch-epoch=009_test_100_samples",
    "Sonics": "sonics_awsaf49_sonics-spectttra-gamma-5s_test_2026-05-14_13-47",
}
OUT = "/net/tscratch/people/plgpkitlo/Audio-XAI/reports/pca_attack.png"

COLORS = {"real": "#2196F3", "fake": "#F44336", "unknown": "#9E9E9E"}
MARKERS = {"original": "o", "adversarial": "X", "delta": "^"}
LABELS_DISPLAY = {"original": "Original", "adversarial": "Adversarial", "delta": "Delta"}


def pool_heatmap(arr: np.ndarray) -> np.ndarray:
    """Reduce variable-length 2D heatmap to fixed feature vector via mean/std/max over time."""
    if arr.ndim == 2:
        return np.concatenate([arr.mean(axis=1), arr.std(axis=1), np.abs(arr).max(axis=1)])
    return arr  # 1D – already fixed size (Sonics)


def load_sample_label(model_dir: str, sample_id: str) -> str:
    root = os.path.join(BASE, model_dir)
    candidates = [f for f in os.listdir(root) if f.endswith(".json") and sample_id in f]
    if candidates:
        with open(os.path.join(root, candidates[0])) as fh:
            data = json.load(fh)
        return data.get("label_str", "unknown")
    return "unknown"


def load_model_data(model_name: str, model_dir: str):
    hpath = os.path.join(BASE, model_dir, "heatmaps")
    samples = sorted(os.listdir(hpath))

    orig_feats, adv_feats, delta_feats, labels = [], [], [], []

    for sample_id in samples:
        sp = os.path.join(hpath, sample_id)
        orig = np.load(os.path.join(sp, "original.npy"))
        adv = np.load(os.path.join(sp, "adversarial.npy"))
        delta = adv - orig

        orig_feats.append(pool_heatmap(orig))
        adv_feats.append(pool_heatmap(adv))
        delta_feats.append(pool_heatmap(delta))
        labels.append(load_sample_label(model_dir, sample_id))

    return (
        np.array(orig_feats),
        np.array(adv_feats),
        np.array(delta_feats),
        labels,
    )


EDGE_COLORS = {
    "original": "navy",
    "adversarial": "darkred",
    "delta": "darkgreen",
}


def plot_model_pca(ax, model_name: str, model_dir: str, show_delta: bool = False):
    print(f"  Loading {model_name}...")
    orig_f, adv_f, delta_f, labels = load_model_data(model_name, model_dir)
    n = len(labels)

    stacks = [orig_f, adv_f] + ([delta_f] if show_delta else [])
    all_feats = np.vstack(stacks)
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_feats)

    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_scaled)

    orig_2d = all_2d[:n]
    adv_2d = all_2d[n : 2 * n]
    delta_2d = all_2d[2 * n :] if show_delta else None

    point_colors = [COLORS.get(lb, COLORS["unknown"]) for lb in labels]

    # Arrows: original → adversarial
    for i in range(n):
        ax.annotate(
            "",
            xy=adv_2d[i],
            xytext=orig_2d[i],
            arrowprops=dict(
                arrowstyle="-|>",
                color="gray",
                alpha=0.20,
                lw=0.8,
                mutation_scale=8,
            ),
        )

    scatter_items = [
        (orig_2d, "original"),
        (adv_2d, "adversarial"),
    ]
    if show_delta:
        scatter_items.append((delta_2d, "delta"))

    for feats_2d, kind in scatter_items:
        ax.scatter(
            feats_2d[:, 0],
            feats_2d[:, 1],
            c=point_colors,
            marker=MARKERS[kind],
            s=55,
            linewidths=0.8,
            edgecolors=EDGE_COLORS[kind],
            alpha=0.8,
            label=LABELS_DISPLAY[kind],
            zorder=3,
        )

    var1, var2 = pca.explained_variance_ratio_
    ax.set_title(f"{model_name}\n(PC1 {var1:.1%}  PC2 {var2:.1%})", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description="PCA of attack heatmaps.")
    parser.add_argument("--delta", action="store_true", help="Include delta (adv−orig) in the plot.")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    subtitle = "Original vs Adversarial vs Delta" if args.delta else "Original vs Adversarial"
    fig.suptitle(f"PCA of Heatmaps: {subtitle}\n(fill = label; edge/shape = type)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (model_name, model_dir) in zip(axes, MODELS.items()):
        plot_model_pca(ax, model_name, model_dir, show_delta=args.delta)

    type_handles = [
        plt.scatter([], [], c="gray", marker=MARKERS["original"], s=60,
                    edgecolors=EDGE_COLORS["original"], linewidths=0.8, label="Original"),
        plt.scatter([], [], c="gray", marker=MARKERS["adversarial"], s=60,
                    edgecolors=EDGE_COLORS["adversarial"], linewidths=0.8, label="Adversarial"),
    ]
    if args.delta:
        type_handles.append(
            plt.scatter([], [], c="gray", marker=MARKERS["delta"], s=60,
                        edgecolors=EDGE_COLORS["delta"], linewidths=0.8, label="Delta (adv−orig)")
        )
    label_handles = [
        mpatches.Patch(color=COLORS["real"], label="real"),
        mpatches.Patch(color=COLORS["fake"], label="fake"),
        plt.Line2D([0], [0], color="gray", alpha=0.4, lw=1.5, label="orig→adv shift"),
    ]

    fig.legend(
        handles=type_handles + label_handles,
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    out_path = OUT.replace(".png", "_with_delta.png") if args.delta else OUT
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()