"""LRP explanation pipeline for trained audio classifiers.

Runs Layer-wise Relevance Propagation over a data split, saves per-sample
heatmaps to disk, and writes spectrograms + LRP overlays to TensorBoard.

Supported models: ast, vggish, wav2vec2
(Sonics is not supported by LRP — use Grad-CAM via predict.py instead.)

Usage:
    python scripts/explain_lrp.py \\
        --config config/predict_ast.yaml \\
        --checkpoint runs/ast/version_3/checkpoints/epoch-epoch=009.ckpt \\
        --data-root audio_xai/data/external \\
        --split test
"""

from __future__ import annotations

import argparse
import csv
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
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_xai.models.ast_binary import ASTBinary, AST_SAMPLE_RATE
from audio_xai.models.vggish_binary import VGGishBinary, VGGISH_SAMPLE_RATE
from audio_xai.models.wav2vec2_binary import Wav2Vec2Binary, WAV2VEC2_SAMPLE_RATE
from audio_xai.xai.lrp import make_lrp


# ── Model loading ─────────────────────────────────────────────────────────────

def _strip_lightning_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_"):
            continue
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def load_model(args) -> tuple[torch.nn.Module, int]:
    """Return (model, sample_rate). Model is on args.device in eval mode."""
    device = args.device

    def _load_ckpt(model):
        if not args.checkpoint:
            return
        print(f"  checkpoint: {args.checkpoint}")
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = _strip_lightning_prefix(ck.get("state_dict", ck))
        model_sd = model.state_dict()
        shape_mm = [k for k, v in sd.items()
                    if k in model_sd and v.shape != model_sd[k].shape]
        if shape_mm:
            print(f"  [warn] skipping shape-mismatched keys: {shape_mm}")
            sd = {k: v for k, v in sd.items() if k not in shape_mm}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] missing  : {missing[:3]}{'…' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"  [warn] unexpected: {unexpected[:3]}{'…' if len(unexpected) > 3 else ''}")

    has_ckpt = bool(args.checkpoint)

    if args.model_type == "ast":
        print(f"Loading ASTBinary… (pretrained={'HuggingFace' if not has_ckpt else 'checkpoint'})")
        model = ASTBinary(pretrained=not has_ckpt, clip_seconds=args.clip_seconds)
        _load_ckpt(model)
        return model.to(device).eval(), AST_SAMPLE_RATE

    if args.model_type == "vggish":
        print(f"Loading VGGishBinary… (pretrained={'yes' if not has_ckpt else 'checkpoint'})")
        model = VGGishBinary()
        _load_ckpt(model)
        return model.to(device).eval(), VGGISH_SAMPLE_RATE

    if args.model_type == "wav2vec2":
        print(f"Loading Wav2Vec2Binary… (pretrained={'HuggingFace' if not has_ckpt else 'checkpoint'})")
        model = Wav2Vec2Binary(pretrained=not has_ckpt)
        _load_ckpt(model)
        return model.to(device).eval(), WAV2VEC2_SAMPLE_RATE

    raise ValueError(f"Unsupported --model-type '{args.model_type}'. "
                     "LRP supports: ast, vggish, wav2vec2")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_root: Path, split: str, n_samples: int | None,
               seed: int = 42) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    split_csv = next((c for c in data_root.glob("*.csv") if c.stem == split), None)

    if split_csv and split_csv.exists():
        with open(split_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = data_root / row["filepath"]
                if not fp.exists():
                    for ext in (".wav", ".mp3", ".flac"):
                        alt = fp.with_suffix(ext)
                        if alt.exists():
                            fp = alt
                            break
                if fp.exists():
                    samples.append((fp, int(row["target"])))
    else:
        for label, subdir in ((0, "real"), (1, "fake")):
            folder = data_root / subdir
            if folder.is_dir():
                for ext in (".wav", ".mp3", ".flac"):
                    for p in sorted(folder.glob(f"*{ext}")):
                        samples.append((p, label))

    if n_samples is not None:
        import random
        rng = random.Random(seed)
        real = [s for s in samples if s[1] == 0]
        fake = [s for s in samples if s[1] == 1]
        rng.shuffle(real); rng.shuffle(fake)
        half = n_samples // 2
        samples = real[:half] + fake[:half]
        rng.shuffle(samples)

    return samples


def load_waveform(path: Path, sample_rate: int, clip_len: int) -> torch.Tensor:
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]
    return torch.from_numpy(y)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _spectrogram_rgb(wav: torch.Tensor, sample_rate: int,
                     max_seconds: float = 10.0) -> np.ndarray:
    """Return [H, W, 3] float32 in [0, 1]."""
    y = wav.cpu().numpy()
    max_samples = int(max_seconds * sample_rate)
    if y.shape[-1] > max_samples:
        y = y[..., :max_samples]
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate,
                                         n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    lo, hi = mel_db.min(), mel_db.max()
    norm = (mel_db - lo) / (hi - lo + 1e-8)
    norm = norm[::-1]  # high freq at top
    return plt.get_cmap("inferno")(norm)[:, :, :3].astype(np.float32)


def _heatmap_rgb(heatmap: np.ndarray, cmap: str = "hot") -> np.ndarray:
    """Normalize a 2-D heatmap → [H, W, 3] float32 in [0, 1]."""
    lo, hi = heatmap.min(), heatmap.max()
    norm = (heatmap - lo) / (hi - lo + 1e-8)
    return plt.get_cmap(cmap)(norm)[:, :, :3].astype(np.float32)


def _to_tb_tensor(rgb: np.ndarray) -> torch.Tensor:
    """[H, W, 3] float32 → [3, H, W] float32 for TensorBoard."""
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def _overlay_figure(
    wav: torch.Tensor,
    heatmap_2d: np.ndarray,
    sample_rate: int,
    label: int,
    pred: int,
    title: str = "",
) -> plt.Figure:
    """Three-panel figure: spectrogram | LRP heatmap | overlay."""
    spec = _spectrogram_rgb(wav, sample_rate)          # [H, W, 3]
    H, W = spec.shape[:2]

    # Resize heatmap to spectrogram dims
    hmap_t = torch.from_numpy(heatmap_2d).unsqueeze(0).unsqueeze(0).float()
    hmap_resized = F.interpolate(hmap_t, size=(H, W), mode="bilinear",
                                 align_corners=False).squeeze().numpy()
    hmap_rgb = _heatmap_rgb(hmap_resized)              # [H, W, 3]
    overlay = np.clip(0.55 * spec + 0.45 * hmap_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    for ax, img, ttl in zip(axes,
                             [spec, hmap_rgb, overlay],
                             ["Spectrogram", "LRP heatmap", "Overlay"]):
        ax.imshow(img, aspect="auto", origin="upper")
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")

    true_lbl = "real" if label == 0 else "fake"
    pred_lbl = "real" if pred == 0 else "fake"
    correct = "✓" if label == pred else "✗"
    fig.suptitle(f"{title}  |  true={true_lbl}  pred={pred_lbl}  {correct}",
                 fontsize=11)
    fig.tight_layout()
    return fig


# Standard output size (H × W pixels) for all mean heatmap PNGs so that
# results from different models (VGGish, AST, …) are directly comparable.
_MEAN_HMAP_H = 128
_MEAN_HMAP_W = 512


def _resize_rgb(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear-resize [H, W, 3] float32 → [h, w, 3]."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()


def _mean_heatmap_figure(mean_hmap: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(_heatmap_rgb(mean_hmap), aspect="auto", origin="upper")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    p.add_argument("--config",       default=None, metavar="YAML")
    p.add_argument("--model-type",   required=False,
                   choices=["ast", "vggish", "wav2vec2"])
    p.add_argument("--checkpoint",   type=Path, default=None)
    p.add_argument("--data-root",    type=Path, required=False)
    p.add_argument("--split",        default="test")
    p.add_argument("--n-samples",    type=int, default=None,
                   help="Cap to N balanced samples (None = all)")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--batch-size",   type=int, default=8,
                   help="Samples per LRP batch (keep small for transformers)")
    p.add_argument("--target-class", type=int, default=None,
                   choices=[0, 1],
                   help="Class to explain: 0=real, 1=fake, None=predicted class")
    p.add_argument("--n-show",       type=int, default=5,
                   help="Number of per-sample overlay figures in TensorBoard")
    p.add_argument("--log-dir",      type=Path, default=Path("runs/lrp"))
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")

    # set_defaults must come after all add_argument calls so it can update action.default
    if _pre_args.config:
        from audio_xai.hparams import predict_defaults
        p.set_defaults(**predict_defaults(_pre_args.config))

    args = p.parse_args()

    if args.model_type is None:
        p.error("--model-type is required (or set model.name in --config YAML)")
    if args.data_root is None:
        p.error("--data-root is required (or set data.root in --config YAML)")

    print(f"Device     : {args.device}")
    print(f"Model      : {args.model_type}")
    print(f"Split      : {args.split}")
    print(f"Target cls : {'argmax (predicted)' if args.target_class is None else args.target_class}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, sample_rate = load_model(args)
    clip_len = int(args.clip_seconds * sample_rate)
    print(f"Sample rate: {sample_rate} Hz  |  clip: {args.clip_seconds}s ({clip_len} samples)")

    # ── 2. Load split ─────────────────────────────────────────────────────────
    samples = load_split(args.data_root, args.split, args.n_samples, seed=args.seed)
    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    print(f"\nSplit '{args.split}': {len(samples)} files  ({n_real} real, {n_fake} fake)")
    if not samples:
        print("No files found — check --data-root and --split.")
        return

    # ── 3. Run LRP ────────────────────────────────────────────────────────────
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else "pretrained"
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{args.model_type}_{ckpt_tag}_{args.split}_{ts}"
    log_path = args.log_dir / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    audio_dir   = log_path / "audio"
    heatmap_dir = log_path / "heatmaps"
    images_dir  = log_path / "images"
    audio_dir.mkdir(exist_ok=True)
    heatmap_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_path))

    all_heatmaps: list[np.ndarray] = []   # [H, W] per sample
    all_labels:   list[int] = []
    all_preds:    list[int] = []
    all_paths:    list[Path] = []
    errors = 0
    t0 = time.time()

    with make_lrp(model) as lrp:
        with tqdm(total=len(samples), desc="LRP", unit="clip",
                  dynamic_ncols=True) as pbar:
            for batch_start in range(0, len(samples), args.batch_size):
                batch = samples[batch_start: batch_start + args.batch_size]
                waveforms, labels, paths = [], [], []
                for path, label in batch:
                    try:
                        waveforms.append(load_waveform(path, sample_rate, clip_len))
                        labels.append(label)
                        paths.append(path)
                    except Exception as e:
                        tqdm.write(f"  [warn] skip {path.name}: {e}")
                        errors += 1

                if not waveforms:
                    pbar.update(len(batch))
                    continue

                x = torch.stack(waveforms).to(args.device)

                try:
                    with torch.enable_grad():
                        heatmap = lrp(x, target_class=args.target_class)  # [B, H, W]
                    with torch.no_grad():
                        logits = model(x)
                    preds = logits.argmax(dim=-1).cpu().tolist()
                except Exception as e:
                    tqdm.write(f"  [warn] LRP failed on batch: {e}")
                    errors += len(waveforms)
                    pbar.update(len(batch))
                    continue

                heatmap_np = heatmap.detach().cpu().numpy()  # [B, H, W]

                for i, (path, label, pred) in enumerate(zip(paths, labels, preds)):
                    all_heatmaps.append(heatmap_np[i])
                    all_labels.append(label)
                    all_preds.append(pred)
                    all_paths.append(path)

                    # ── audio ────────────────────────────────────────────────
                    sample_audio_dir = audio_dir / path.stem
                    sample_audio_dir.mkdir(exist_ok=True)
                    torchaudio.save(
                        str(sample_audio_dir / "original.wav"),
                        waveforms[i].unsqueeze(0), sample_rate,
                    )

                    # ── heatmap (.npy) ───────────────────────────────────────
                    sample_heatmap_dir = heatmap_dir / path.stem
                    sample_heatmap_dir.mkdir(exist_ok=True)
                    np.save(str(sample_heatmap_dir / "lrp.npy"), heatmap_np[i])

                    # ── PNG images ───────────────────────────────────────────
                    sample_images_dir = images_dir / path.stem
                    sample_images_dir.mkdir(exist_ok=True)

                    wav_cpu = waveforms[i]
                    spec    = _spectrogram_rgb(wav_cpu, sample_rate)   # [H, W, 3]
                    hmap_2d = heatmap_np[i]
                    hmap_t  = torch.from_numpy(hmap_2d).unsqueeze(0).unsqueeze(0).float()
                    hmap_rs = F.interpolate(hmap_t, size=spec.shape[:2],
                                            mode="bilinear", align_corners=False
                                            ).squeeze().numpy()
                    hmap    = _heatmap_rgb(hmap_rs)                    # [H, W, 3]
                    overlay = np.clip(0.55 * spec + 0.45 * hmap, 0, 1)

                    plt.imsave(str(sample_images_dir / "spectrogram.png"), spec)
                    plt.imsave(str(sample_images_dir / "heatmap.png"),     hmap)
                    plt.imsave(str(sample_images_dir / "overlay.png"),     overlay)

                    # 3-panel figure saved as a single PNG
                    true_lbl = "real" if label == 0 else "fake"
                    pred_lbl = "real" if pred  == 0 else "fake"
                    correct  = "OK" if label == pred else "WRONG"
                    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
                    for ax, img, ttl in zip(axes,
                                            [spec, hmap, overlay],
                                            ["Spectrogram", "LRP heatmap", "Overlay"]):
                        ax.imshow(img, aspect="auto", origin="upper")
                        ax.set_title(ttl, fontsize=10)
                        ax.axis("off")
                    fig.suptitle(
                        f"{args.model_type} / {path.stem}  |  "
                        f"true={true_lbl}  pred={pred_lbl}  [{correct}]",
                        fontsize=11,
                    )
                    fig.tight_layout()
                    fig.savefig(str(sample_images_dir / "panel.png"),
                                dpi=120, bbox_inches="tight")
                    plt.close(fig)

                pbar.update(len(batch))
                pbar.set_postfix(errors=errors)

    elapsed = time.time() - t0
    print(f"\nLRP done: {len(all_heatmaps)} samples in {elapsed:.1f}s  "
          f"({errors} skipped)")

    # ── 4. TensorBoard — per-sample overlays ──────────────────────────────────
    n_show = min(args.n_show, len(all_heatmaps))
    for i in range(n_show):
        path, label, pred = all_paths[i], all_labels[i], all_preds[i]
        try:
            wav = load_waveform(path, sample_rate, clip_len)
        except Exception:
            continue
        fig = _overlay_figure(
            wav, all_heatmaps[i], sample_rate, label, pred,
            title=f"{args.model_type} / {path.stem}"
        )
        writer.add_figure(f"lrp/{path.stem}", fig, 0)
        plt.close(fig)

        # Also write spectrogram and heatmap as separate images
        spec_rgb = _spectrogram_rgb(wav, sample_rate)
        hmap_rgb = _heatmap_rgb(all_heatmaps[i])
        writer.add_image(f"spectrogram/{path.stem}", _to_tb_tensor(spec_rgb), 0)
        writer.add_image(f"heatmap/{path.stem}",    _to_tb_tensor(hmap_rgb), 0)

    # ── 5. Mean heatmaps per class ────────────────────────────────────────────
    for cls_id, cls_name in [(0, "real"), (1, "fake")]:
        hmaps = [h for h, l in zip(all_heatmaps, all_labels) if l == cls_id]
        if not hmaps:
            continue
        # Resize all to the first one's shape before averaging
        ref_shape = hmaps[0].shape
        resized = []
        for h in hmaps:
            if h.shape == ref_shape:
                resized.append(h)
            else:
                t = torch.from_numpy(h).unsqueeze(0).unsqueeze(0).float()
                t = F.interpolate(t, size=ref_shape, mode="bilinear", align_corners=False)
                resized.append(t.squeeze().numpy())
        mean_hmap = np.mean(resized, axis=0)
        np.save(str(log_path / f"mean_heatmap_{cls_name}.npy"), mean_hmap)
        plt.imsave(
            str(images_dir / f"mean_heatmap_{cls_name}.png"),
            _resize_rgb(_heatmap_rgb(mean_hmap), _MEAN_HMAP_H, _MEAN_HMAP_W),
        )
        fig = _mean_heatmap_figure(mean_hmap, f"Mean LRP relevance — {cls_name} ({len(hmaps)} samples)")
        fig.savefig(str(images_dir / f"mean_heatmap_{cls_name}_figure.png"),
                    dpi=120, bbox_inches="tight")
        writer.add_figure(f"lrp_mean/{cls_name}", fig, 0)
        plt.close(fig)
        writer.add_image(f"heatmap_mean/{cls_name}", _to_tb_tensor(_heatmap_rgb(mean_hmap)), 0)

    # ── 6. Accuracy + text summary ────────────────────────────────────────────
    if all_labels and all_preds:
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        print(f"Accuracy on explained samples: {acc:.4f}")

        summary_md = (
            f"**LRP Explanation Summary**\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| model | {args.model_type} |\n"
            f"| checkpoint | {args.checkpoint or 'pretrained'} |\n"
            f"| split | {args.split} |\n"
            f"| n_samples | {len(all_labels)} |\n"
            f"| target_class | {args.target_class if args.target_class is not None else 'predicted'} |\n"
            f"| clip_seconds | {args.clip_seconds} |\n"
            f"| accuracy | {acc:.4f} |\n"
            f"| errors | {errors} |\n"
        )
        writer.add_text("summary", summary_md, 0)

    # ── 7. Save JSON summary ──────────────────────────────────────────────────
    summary = {
        "model_type":   args.model_type,
        "checkpoint":   str(args.checkpoint) if args.checkpoint else None,
        "split":        args.split,
        "n_samples":    len(all_labels),
        "target_class": args.target_class,
        "clip_seconds": args.clip_seconds,
        "errors":       errors,
        "per_sample": [
            {"file": str(all_paths[i].name), "label": all_labels[i],
             "pred": all_preds[i], "heatmap": f"{all_paths[i].stem}.npy"}
            for i in range(len(all_labels))
        ],
    }
    (log_path / "lrp_summary.json").write_text(json.dumps(summary, indent=2))

    writer.close()
    print(f"\nAudio    (.wav) → {audio_dir}/  ({len(all_heatmaps)} subdirs)")
    print(f"  per sample : <stem>/original.wav")
    print(f"Heatmaps (.npy) → {heatmap_dir}/  ({len(all_heatmaps)} subdirs)")
    print(f"  per sample : <stem>/lrp.npy")
    print(f"Images   (.png) → {images_dir}/")
    print(f"  per sample : <stem>/spectrogram.png  heatmap.png  overlay.png  panel.png")
    print(f"  mean       : mean_heatmap_real.png  mean_heatmap_fake.png")
    print(f"TensorBoard → {log_path}")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()