"""Average heatmaps per label (real/fake) and per attack type.

For each explainer (GradCAM, LRP):
  - Collects all heatmaps across available runs
  - Groups by true label (real/fake) × audio type (original/adversarial)
  - Groups by attack type × audio type (original/adversarial)
  - Normalises each heatmap to [0,1], resizes to TARGET_SHAPE, then averages

Outputs:
  reports/avg_heatmap_gradcam.png          — per label
  reports/avg_heatmap_lrp.png             — per label
  reports/avg_heatmap_gradcam_per_attack.png  — per attack
  reports/avg_heatmap_lrp_per_attack.png      — per attack
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS_ROOT   = Path(__file__).resolve().parent.parent / "runs"
OUT_DIR     = Path(__file__).resolve().parent.parent / "reports"
TARGET_SHAPE = (64, 512)   # (freq, time) — common canvas for all models

# GradCAM exists for all 9 runs.
# Perceptual attack saves original.npy / adversarial.npy.
# PGD and X-Shift saves saliency_original.npy / saliency_adversarial.npy.
GRADCAM_RUNS = [
    "attack/ast_epoch=4-step=9740_test_100_samples",
    "attack/vggish_epoch-epoch=009_test_100_samples",
    "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    "pgd_attack/pgd_ast_epoch=4-step=9740_test_100_samples",
    "pgd_attack/pgd_vggish_epoch-epoch=009_test_100_samples",
    "pgd_attack/pgd_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    "xshift_attack/xshift_ast_epoch=4-step=9740_test_100_samples_shift",
    "xshift_attack/xshift_vggish_epoch-epoch=009_test_100_samples_shift",
    "xshift_attack/xshift_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples_shift",
]

# LRP exists for all 9 runs
LRP_RUNS = [
    "attack/ast_epoch=4-step=9740_test_100_samples",
    "attack/vggish_epoch-epoch=009_test_100_samples",
    "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    "pgd_attack/pgd_ast_epoch=4-step=9740_test_100_samples",
    "pgd_attack/pgd_vggish_epoch-epoch=009_test_100_samples",
    "pgd_attack/pgd_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    "xshift_attack/xshift_ast_epoch=4-step=9740_test_100_samples_shift",
    "xshift_attack/xshift_vggish_epoch-epoch=009_test_100_samples_shift",
    "xshift_attack/xshift_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples_shift",
]

# Runs grouped by attack type (used for both LRP and GradCAM per-attack plots)
RUNS_BY_ATTACK: dict[str, list[str]] = {
    "Perceptual": [
        "attack/ast_epoch=4-step=9740_test_100_samples",
        "attack/vggish_epoch-epoch=009_test_100_samples",
        "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    ],
    "PGD": [
        "pgd_attack/pgd_ast_epoch=4-step=9740_test_100_samples",
        "pgd_attack/pgd_vggish_epoch-epoch=009_test_100_samples",
        "pgd_attack/pgd_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    ],
    "X-Shift": [
        "xshift_attack/xshift_ast_epoch=4-step=9740_test_100_samples_shift",
        "xshift_attack/xshift_vggish_epoch-epoch=009_test_100_samples_shift",
        "xshift_attack/xshift_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples_shift",
    ],
}

# Filename pairs per subdir (GradCAM filenames differ by run type)
GRADCAM_FILENAMES: dict[str, tuple[str, str]] = {
    "attack":        ("original.npy",          "adversarial.npy"),
    "pgd_attack":    ("saliency_original.npy",  "saliency_adversarial.npy"),
    "xshift_attack": ("saliency_original.npy",  "saliency_adversarial.npy"),
}


def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D (or 1-D) heatmap to shape using bilinear interpolation."""
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]          # (1, N)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    t = F.interpolate(t, size=shape, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def _gradcam_fnames(rel: str) -> tuple[str, str]:
    """Return (orig_fname, adv_fname) for a GradCAM run based on its attack subdir."""
    attack_type = rel.split("/")[0]
    return GRADCAM_FILENAMES.get(attack_type, ("original.npy", "adversarial.npy"))


def collect(
    run_dirs: list[str],
    subdir: str,
    orig_fname: str | None,
    adv_fname: str | None,
    *,
    per_run_fnames: bool = False,
) -> dict[tuple[str, str], list[np.ndarray]]:
    """Return {(label, 'original'|'adversarial'): [resized_heatmaps]}.

    If per_run_fnames=True, ignore orig_fname/adv_fname and look up filenames
    from GRADCAM_FILENAMES based on each run's attack subdirectory.
    """
    buckets: dict[tuple[str, str], list[np.ndarray]] = {
        ("real", "original"): [], ("real", "adversarial"): [],
        ("fake", "original"): [], ("fake", "adversarial"): [],
    }

    for rel in run_dirs:
        run_dir = RUNS_ROOT / rel
        hmap_dir = run_dir / subdir
        if not hmap_dir.exists():
            print(f"  [skip] {subdir}/ missing in {rel}")
            continue

        of, af = _gradcam_fnames(rel) if per_run_fnames else (orig_fname, adv_fname)

        # Load label info from sample JSONs
        label_map = {}
        for jf in run_dir.glob("sample_*.json"):
            d = json.loads(jf.read_text())
            stem = d.get("stem", "")
            if stem:
                label_map[stem] = d.get("label_str", "")

        for stem_dir in hmap_dir.iterdir():
            if not stem_dir.is_dir():
                continue
            label = label_map.get(stem_dir.name, "")
            if label not in ("real", "fake"):
                continue

            for key, fname in [("original", of), ("adversarial", af)]:
                fpath = stem_dir / fname
                if not fpath.exists():
                    continue
                arr = np.load(str(fpath)).astype(np.float32)
                arr = _normalise(arr)
                arr = _resize(arr, TARGET_SHAPE)
                buckets[(label, key)].append(arr)

    return buckets


def average_buckets(
    buckets: dict[tuple[str, str], list[np.ndarray]],
) -> dict[tuple[str, str], np.ndarray | None]:
    return {
        k: np.mean(v, axis=0) if v else None
        for k, v in buckets.items()
    }


def save_figure(
    avgs: dict[tuple[str, str], np.ndarray | None],
    title: str,
    out_path: Path,
    n_per_cell: dict[tuple[str, str], int],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    rows   = ["real", "fake"]
    cols   = ["original", "adversarial"]
    cmaps  = {"real": "Blues", "fake": "Reds"}

    for r, label in enumerate(rows):
        for c, atype in enumerate(cols):
            ax  = axes[r, c]
            arr = avgs.get((label, atype))
            n   = n_per_cell.get((label, atype), 0)

            if arr is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=40, transform=ax.transAxes)
                ax.axis("off")
            else:
                vmin = np.percentile(arr, 2)
                vmax = np.percentile(arr, 98)
                im = ax.imshow(
                    arr, aspect="auto", origin="upper",
                    cmap=cmaps[label], vmin=vmin, vmax=vmax,
                )
                plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

            ax.set_title(
                f"{label.capitalize()} — {atype}  (n={n})",
                fontsize=25, fontweight="bold",
            )
            ax.set_xlabel("time →", fontsize=20)
            ax.set_ylabel("freq →", fontsize=20)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def collect_per_attack(
    runs_by_attack: dict[str, list[str]],
    subdir: str,
    orig_fname: str | None,
    adv_fname: str | None,
    *,
    per_run_fnames: bool = False,
) -> dict[tuple[str, str], list[np.ndarray]]:
    """Return {(attack_name, 'original'|'adversarial'): [resized_heatmaps]}.

    Averages across all models and real/fake labels within each attack.
    If per_run_fnames=True, filenames are looked up from GRADCAM_FILENAMES.
    """
    buckets: dict[tuple[str, str], list[np.ndarray]] = {}
    for attack_name, run_dirs in runs_by_attack.items():
        for atype in ("original", "adversarial"):
            buckets[(attack_name, atype)] = []

    for attack_name, run_dirs in runs_by_attack.items():
        for rel in run_dirs:
            run_dir = RUNS_ROOT / rel
            hmap_dir = run_dir / subdir
            if not hmap_dir.exists():
                print(f"  [skip] {subdir}/ missing in {rel}")
                continue

            of, af = _gradcam_fnames(rel) if per_run_fnames else (orig_fname, adv_fname)

            for stem_dir in hmap_dir.iterdir():
                if not stem_dir.is_dir():
                    continue
                for atype, fname in [("original", of), ("adversarial", af)]:
                    fpath = stem_dir / fname
                    if not fpath.exists():
                        continue
                    arr = np.load(str(fpath)).astype(np.float32)
                    arr = _normalise(arr)
                    arr = _resize(arr, TARGET_SHAPE)
                    buckets[(attack_name, atype)].append(arr)

    return buckets


def save_figure_per_attack(
    avgs: dict[tuple[str, str], np.ndarray | None],
    title: str,
    out_path: Path,
    n_per_cell: dict[tuple[str, str], int],
    attack_names: list[str],
) -> None:
    n_rows = len(attack_names)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    cmap = "hot"
    for r, attack in enumerate(attack_names):
        for c, atype in enumerate(("original", "adversarial")):
            ax  = axes[r, c]
            arr = avgs.get((attack, atype))
            n   = n_per_cell.get((attack, atype), 0)

            if arr is None or n == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=40, transform=ax.transAxes)
                ax.axis("off")
            else:
                vmin = np.percentile(arr, 2)
                vmax = np.percentile(arr, 98)
                im = ax.imshow(arr, aspect="auto", origin="upper",
                               cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

            ax.set_title(f"{attack} — {atype}  (n={n})",
                         fontsize=25, fontweight="bold")
            ax.set_xlabel("time →", fontsize=20)
            ax.set_ylabel("freq →", fontsize=20)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

    fig.suptitle(title, fontsize=32, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── GradCAM ──────────────────────────────────────────────────────────────
    print("Collecting GradCAM heatmaps…")
    gcam_buckets = collect(
        GRADCAM_RUNS, "heatmaps", None, None, per_run_fnames=True,
    )
    gcam_counts = {k: len(v) for k, v in gcam_buckets.items()}
    print("  counts:", {f"{k[0]}/{k[1]}": v for k, v in gcam_counts.items()})
    gcam_avgs = average_buckets(gcam_buckets)
    save_figure(
        gcam_avgs,
        title=f"GradCAM — average heatmap per label & audio type",
        out_path=OUT_DIR / "avg_heatmap_gradcam.png",
        n_per_cell=gcam_counts,
    )

    # ── LRP ──────────────────────────────────────────────────────────────────
    print("Collecting LRP heatmaps…")
    lrp_buckets = collect(
        LRP_RUNS, "lrp", "lrp_original.npy", "lrp_adversarial.npy"
    )
    lrp_counts = {k: len(v) for k, v in lrp_buckets.items()}
    print("  counts:", {f"{k[0]}/{k[1]}": v for k, v in lrp_counts.items()})
    lrp_avgs = average_buckets(lrp_buckets)
    save_figure(
        lrp_avgs,
        title=f"LRP — average heatmap per label & audio type",
        out_path=OUT_DIR / "avg_heatmap_lrp.png",
        n_per_cell=lrp_counts,
    )

    # ── GradCAM per attack ────────────────────────────────────────────────────
    print("Collecting GradCAM heatmaps per attack…")
    gcam_attack_buckets = collect_per_attack(
        RUNS_BY_ATTACK, "heatmaps", None, None, per_run_fnames=True,
    )
    gcam_attack_counts = {k: len(v) for k, v in gcam_attack_buckets.items()}
    print("  counts:", {f"{k[0]}/{k[1]}": v for k, v in gcam_attack_counts.items()})
    gcam_attack_avgs = average_buckets(gcam_attack_buckets)
    save_figure_per_attack(
        gcam_attack_avgs,
        title=f"GradCAM — average heatmap per attack\n"
              f"(3 models, all labels, resized to {TARGET_SHAPE})",
        out_path=OUT_DIR / "avg_heatmap_gradcam_per_attack.png",
        n_per_cell=gcam_attack_counts,
        attack_names=list(RUNS_BY_ATTACK.keys()),
    )

    # ── LRP per attack ────────────────────────────────────────────────────────
    print("Collecting LRP heatmaps per attack…")
    lrp_attack_buckets = collect_per_attack(
        RUNS_BY_ATTACK, "lrp", "lrp_original.npy", "lrp_adversarial.npy",
    )
    lrp_attack_counts = {k: len(v) for k, v in lrp_attack_buckets.items()}
    print("  counts:", {f"{k[0]}/{k[1]}": v for k, v in lrp_attack_counts.items()})
    lrp_attack_avgs = average_buckets(lrp_attack_buckets)
    save_figure_per_attack(
        lrp_attack_avgs,
        title=f"LRP — average heatmap per attack\n"
              f"(3 models, all labels, resized to {TARGET_SHAPE})",
        out_path=OUT_DIR / "avg_heatmap_lrp_per_attack.png",
        n_per_cell=lrp_attack_counts,
        attack_names=list(RUNS_BY_ATTACK.keys()),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
