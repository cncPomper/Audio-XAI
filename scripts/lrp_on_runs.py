"""Generate LRP heatmaps for all audio files in attack run directories.

For each run directory, reads results.json to determine model type and
checkpoint, then computes LRP maps for original.wav and adversarial.wav
in every audio/<stem>/ subdirectory.

Outputs go to <run_dir>/lrp/<stem>/:
  lrp_original.npy        raw heatmap (float32)
  lrp_adversarial.npy     raw heatmap (float32)
  spectrogram_original.png
  spectrogram_adversarial.png
  heatmap_original.png
  heatmap_adversarial.png
  overlay_original.png
  overlay_adversarial.png
  panel.png               2×3 grid (spec | heatmap | overlay) × (orig | adv)

Supports AST, VGGish, and Sonics (SpecTTTra) models.

Usage:
    python scripts/lrp_on_runs.py [--device cuda] [--batch-size 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from audio_xai.models.ast_binary import ASTBinary, AST_SAMPLE_RATE
from audio_xai.models.sonics_wrapper import SonicsWrapper
from audio_xai.models.vggish_binary import VGGishBinary, VGGISH_SAMPLE_RATE
from audio_xai.xai.lrp import make_lrp

SONICS_SAMPLE_RATE = 16_000


# ── Run directories to process ────────────────────────────────────────────────

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"

RUN_DIRS: list[str] = [
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


# ── Model loading ─────────────────────────────────────────────────────────────

def _strip_lightning_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_"):
            continue
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def load_model(model_type: str, checkpoint: str | None,
               clip_seconds: float, device: str, model_id: str | None = None):
    """Return (model, sample_rate). Model is on device in eval mode."""
    def _load_ckpt(model):
        if not checkpoint:
            return
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = _strip_lightning_prefix(ck.get("state_dict", ck))
        model_sd = model.state_dict()
        shape_mm = [k for k, v in sd.items()
                    if k in model_sd and v.shape != model_sd[k].shape]
        if shape_mm:
            print(f"  [warn] skipping shape-mismatched keys: {shape_mm}")
            sd = {k: v for k, v in sd.items() if k not in shape_mm}
        model.load_state_dict(sd, strict=False)

    if model_type == "ast":
        model = ASTBinary(pretrained=not checkpoint, clip_seconds=clip_seconds)
        _load_ckpt(model)
        return model.to(device).eval(), AST_SAMPLE_RATE

    if model_type == "vggish":
        model = VGGishBinary()
        _load_ckpt(model)
        return model.to(device).eval(), VGGISH_SAMPLE_RATE

    if model_type == "sonics":
        from sonics import HFAudioClassifier
        if not model_id:
            raise ValueError("model_id required for model_type='sonics'")
        print(f"  Loading Sonics model: {model_id}")
        raw = HFAudioClassifier.from_pretrained(model_id, map_location="cpu")
        wrapped = SonicsWrapper(raw).to(device).eval()
        return wrapped, SONICS_SAMPLE_RATE

    raise ValueError(f"No LRP implementation for model_type={model_type!r}")


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_waveform(path: Path, sample_rate: int, clip_len: int) -> torch.Tensor:
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]
    return torch.from_numpy(y)


def load_waveform_full(path: Path, sample_rate: int, clip_len: int) -> torch.Tensor:
    """Load full audio without truncation; pad to at least clip_len."""
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    return torch.from_numpy(y)


def chunk_waveform(wav: torch.Tensor, clip_len: int) -> list[torch.Tensor]:
    """Split waveform into non-overlapping clip_len chunks, padding the last."""
    chunks = []
    total = wav.shape[0]
    for start in range(0, total, clip_len):
        chunk = wav[start: start + clip_len]
        if chunk.shape[0] < clip_len:
            chunk = F.pad(chunk, (0, clip_len - chunk.shape[0]))
        chunks.append(chunk)
    return chunks


def lrp_chunked(
    lrp_fn,
    wav: torch.Tensor,
    clip_len: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run LRP on full waveform by processing clip_len chunks and stitching."""
    chunks = chunk_waveform(wav, clip_len)
    heatmaps: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = torch.stack(chunks[i: i + batch_size]).to(device)
        with torch.enable_grad():
            h = lrp_fn(batch).detach().cpu().numpy()  # [B, freq, time]
        for j in range(h.shape[0]):
            heatmaps.append(h[j])
    # heatmaps: list of [freq, time_per_chunk] → stitch along time axis
    return np.concatenate(heatmaps, axis=-1)  # [freq, time_total]


# ── Visualisation ─────────────────────────────────────────────────────────────

def _spectrogram_gray(wav: torch.Tensor, sample_rate: int) -> np.ndarray:
    """[H, W, 3] float32 grayscale spectrogram (to match the reference panel)."""
    y = wav.cpu().numpy()
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate,
                                         n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    lo, hi = mel_db.min(), mel_db.max()
    norm = (mel_db - lo) / (hi - lo + 1e-8)
    norm = norm[::-1]  # high freq at top
    gray = np.stack([norm, norm, norm], axis=-1).astype(np.float32)
    return gray


def _heatmap_rgb(heatmap: np.ndarray, h: int, w: int,
                 cmap: str = "hot") -> np.ndarray:
    """Resize heatmap to (h, w) and colorize → [H, W, 3] float32."""
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    norm_arr = t.squeeze().numpy()
    lo, hi = norm_arr.min(), norm_arr.max()
    norm_arr = (norm_arr - lo) / (hi - lo + 1e-8)
    return plt.get_cmap(cmap)(norm_arr)[:, :, :3].astype(np.float32)


def save_panel(
    wav_orig: torch.Tensor,
    wav_adv: torch.Tensor,
    hmap_orig: np.ndarray,
    hmap_adv: np.ndarray,
    sample_rate: int,
    stem: str,
    label: int,
    pred_orig: int,
    pred_adv: int,
    out_dir: Path,
) -> None:
    """Save per-file PNGs and the 2×3 panel image."""
    spec_o = _spectrogram_gray(wav_orig, sample_rate)
    spec_a = _spectrogram_gray(wav_adv,  sample_rate)
    H, W = spec_o.shape[:2]

    hmap_o = _heatmap_rgb(hmap_orig, H, W)
    hmap_a = _heatmap_rgb(hmap_adv,  H, W)

    ov_o = np.clip(0.55 * spec_o + 0.45 * hmap_o, 0, 1)
    ov_a = np.clip(0.55 * spec_a + 0.45 * hmap_a, 0, 1)

    # individual PNGs
    plt.imsave(str(out_dir / "spectrogram_original.png"),    spec_o)
    plt.imsave(str(out_dir / "spectrogram_adversarial.png"), spec_a)
    plt.imsave(str(out_dir / "heatmap_original.png"),        hmap_o)
    plt.imsave(str(out_dir / "heatmap_adversarial.png"),     hmap_a)
    plt.imsave(str(out_dir / "overlay_original.png"),        ov_o)
    plt.imsave(str(out_dir / "overlay_adversarial.png"),     ov_a)

    # 2×3 panel
    true_lbl = "real" if label == 0 else "fake"
    po = "real" if pred_orig == 0 else "fake"
    pa = "real" if pred_adv  == 0 else "fake"

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    data = [
        (spec_o, hmap_o, ov_o,
         "Original spectrogram", "Original LRP", "Original overlay"),
        (spec_a, hmap_a, ov_a,
         "Adversarial spectrogram", "Adversarial LRP", "Adversarial overlay"),
    ]
    for row, (s, h, o, ts, th, to) in enumerate(data):
        for col, (img, ttl) in enumerate([(s, ts), (h, th), (o, to)]):
            axes[row, col].imshow(img, aspect="auto", origin="upper")
            axes[row, col].set_title(ttl, fontsize=9)
            axes[row, col].axis("off")

    fig.suptitle(
        f"{stem}  |  true={true_lbl}  pred_orig={po}  pred_adv={pa}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "panel.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Sample metadata loader ────────────────────────────────────────────────────

def load_sample_meta(run_dir: Path) -> dict[str, dict]:
    """Return {stem: {label, pred_orig, pred_adv, _path}} from per-sample JSON files."""
    meta: dict[str, dict] = {}
    for jf in run_dir.glob("sample_*.json"):
        try:
            d = json.loads(jf.read_text())
            stem = d.get("stem", "")
            if stem:
                meta[stem] = {
                    "label":     d.get("label", -1),
                    "pred_orig": d.get("pred_orig", -1),
                    "pred_adv":  d.get("pred_adv",  -1),
                    "_path":     jf,
                }
        except Exception:
            pass
    return meta


# ── LRP metrics ──────────────────────────────────────────────────────────────

def _lrp_flatten_normalize(h: np.ndarray) -> np.ndarray:
    flat = h.flatten().astype(np.float32)
    norm = np.linalg.norm(flat) + 1e-8
    return flat / norm


def lrp_cos_sim(hmap_a: np.ndarray, hmap_b: np.ndarray) -> float:
    return float(np.dot(_lrp_flatten_normalize(hmap_a), _lrp_flatten_normalize(hmap_b)))


def lrp_top10_overlap(hmap_a: np.ndarray, hmap_b: np.ndarray, k_frac: float = 0.1) -> float:
    flat_a = hmap_a.flatten()
    flat_b = hmap_b.flatten()
    k = max(1, int(k_frac * flat_a.size))
    idx_a = set(np.argpartition(flat_a, -k)[-k:].tolist())
    idx_b = set(np.argpartition(flat_b, -k)[-k:].tolist())
    inter = len(idx_a & idx_b)
    union = len(idx_a | idx_b)
    return inter / union if union > 0 else 0.0


def update_sample_json(json_path: Path, updates: dict) -> None:
    d = json.loads(json_path.read_text())
    d.update(updates)
    json_path.write_text(json.dumps(d, indent=2))


# ── Per-run processing ────────────────────────────────────────────────────────

def process_run(run_dir: Path, device: str, batch_size: int) -> None:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        print(f"  [skip] no results.json in {run_dir}")
        return

    run_cfg = json.loads(results_file.read_text()).get("run", {})
    model_type  = run_cfg.get("model_type", "")
    checkpoint  = run_cfg.get("checkpoint", None)
    model_id    = run_cfg.get("model_id", None)
    clip_secs   = float(run_cfg.get("clip_seconds", 5.0))

    if model_type not in ("ast", "vggish", "sonics"):
        print(f"  [skip] Unknown model_type={model_type!r} in {run_dir.name}")
        return

    print(f"\n{'='*70}")
    print(f"Run  : {run_dir.name}")
    print(f"Model: {model_type}  checkpoint: {checkpoint or model_id}")

    model, sample_rate = load_model(model_type, checkpoint, clip_secs, device, model_id=model_id)
    clip_len = int(clip_secs * sample_rate)
    sample_meta = load_sample_meta(run_dir)

    audio_root = run_dir / "audio"
    if not audio_root.is_dir():
        print(f"  [skip] no audio/ directory in {run_dir}")
        return

    stems = sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    lrp_root = run_dir / "lrp"
    lrp_root.mkdir(exist_ok=True)

    errors = 0
    t0 = time.time()

    with make_lrp(model) as lrp_fn:
        for stem in tqdm(stems, desc=run_dir.name, unit="stem"):
            orig_path = audio_root / stem / "original.wav"
            adv_path  = audio_root / stem / "adversarial.wav"
            if not orig_path.exists() or not adv_path.exists():
                tqdm.write(f"  [warn] missing audio for {stem}")
                errors += 1
                continue

            try:
                wav_orig = load_waveform_full(orig_path, sample_rate, clip_len)
                wav_adv  = load_waveform_full(adv_path,  sample_rate, clip_len)
            except Exception as e:
                tqdm.write(f"  [warn] load error {stem}: {e}")
                errors += 1
                continue

            try:
                hmap_orig = lrp_chunked(lrp_fn, wav_orig, clip_len, device, batch_size)
                hmap_adv  = lrp_chunked(lrp_fn, wav_adv,  clip_len, device, batch_size)
            except Exception as e:
                tqdm.write(f"  [warn] LRP failed for {stem}: {e}")
                errors += 1
                continue

            out_dir = lrp_root / stem
            out_dir.mkdir(exist_ok=True)

            np.save(str(out_dir / "lrp_original.npy"),    hmap_orig)
            np.save(str(out_dir / "lrp_adversarial.npy"), hmap_adv)

            meta = sample_meta.get(stem, {})
            save_panel(
                wav_orig, wav_adv,
                hmap_orig, hmap_adv,
                sample_rate, stem,
                label=meta.get("label", -1),
                pred_orig=meta.get("pred_orig", -1),
                pred_adv=meta.get("pred_adv", -1),
                out_dir=out_dir,
            )

            json_path = meta.get("_path")
            if json_path is not None:
                try:
                    update_sample_json(json_path, {
                        "lrp_cos_sim":      round(lrp_cos_sim(hmap_orig, hmap_adv), 6),
                        "lrp_top10_overlap": round(lrp_top10_overlap(hmap_orig, hmap_adv), 6),
                    })
                except Exception as e:
                    tqdm.write(f"  [warn] JSON update failed for {stem}: {e}")

    elapsed = time.time() - t0
    n_done = len(stems) - errors
    print(f"  Done: {n_done}/{len(stems)} samples in {elapsed:.1f}s  "
          f"({errors} skipped)  → {lrp_root}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("runs", nargs="*", metavar="RUN_DIR",
                   help="Override run directories (absolute or relative to runs/). "
                        "Default: the 9 hardcoded attack directories.")
    args = p.parse_args()

    if args.runs:
        dirs = [Path(r) if Path(r).is_absolute() else RUNS_ROOT / r
                for r in args.runs]
    else:
        dirs = [RUNS_ROOT / r for r in RUN_DIRS]

    print(f"Device    : {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Runs      : {len(dirs)}")

    for run_dir in dirs:
        if not run_dir.is_dir():
            print(f"[skip] not found: {run_dir}")
            continue
        process_run(run_dir, args.device, args.batch_size)

    print("\nAll done.")


if __name__ == "__main__":
    main()
