"""Real/fake audio prediction for AST, VGGish, and Sonics.

Batch inference over the full split with accuracy/F1/AUC metrics.

Usage examples:
    python scripts/predict.py --model-type sonics \\
        --model-id awsaf49/sonics-spectttra-gamma-120s \\
        --clip-seconds 120.0 --data-root audio_xai/data/external

    python scripts/predict.py --model-type ast \\
        --checkpoint runs/ast/version_2/checkpoints/epoch=2-step=1500.ckpt \\
        --data-root audio_xai/data/external
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from audio_xai.models.ast_binary import ASTBinary, AST_SAMPLE_RATE
from audio_xai.models.vggish_binary import VGGishBinary, VGGISH_SAMPLE_RATE


# ── Model loading ─────────────────────────────────────────────────────────────

def _strip_lightning_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_"):
            continue
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def load_model(args) -> tuple[torch.nn.Module, int]:
    """Return (model, sample_rate)."""
    device = args.device

    if args.model_type == "sonics":
        from sonics import HFAudioClassifier
        print(f"Loading Sonics model: {args.model_id}")
        model = HFAudioClassifier.from_pretrained(args.model_id, map_location=device)
        model = model.to(device).eval()
        print(f"  input_shape={model.input_shape}  n_classes={model.num_classes}")
        return model, args.sample_rate

    if args.model_type == "ast":
        has_ckpt = bool(args.checkpoint)
        print(f"Loading ASTBinary… (pretrained={'HuggingFace' if not has_ckpt else 'checkpoint'})")
        model = ASTBinary(pretrained=not has_ckpt, clip_seconds=args.clip_seconds)
        if args.checkpoint:
            print(f"  fine-tuned weights: {args.checkpoint}")
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            sd = _strip_lightning_prefix(ck.get("state_dict", ck))
            model_sd = model.state_dict()
            shape_mismatch = [k for k, v in sd.items() if k in model_sd and v.shape != model_sd[k].shape]
            if shape_mismatch:
                print(f"  [warn] skipping shape-mismatched keys (checkpoint clip_seconds differs from --clip-seconds {args.clip_seconds}): {shape_mismatch}")
                sd = {k: v for k, v in sd.items() if k not in shape_mismatch}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  [warn] missing: {missing[:3]}{'…' if len(missing)>3 else ''}")
            if unexpected:
                print(f"  [warn] unexpected: {unexpected[:3]}{'…' if len(unexpected)>3 else ''}")
        model = model.to(device).eval()
        return model, AST_SAMPLE_RATE

    if args.model_type == "vggish":
        print("Loading VGGishBinary…")
        model = VGGishBinary()
        if args.checkpoint:
            print(f"  fine-tuned weights: {args.checkpoint}")
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            sd = _strip_lightning_prefix(ck.get("state_dict", ck))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  [warn] missing: {missing[:3]}{'…' if len(missing)>3 else ''}")
            if unexpected:
                print(f"  [warn] unexpected: {unexpected[:3]}{'…' if len(unexpected)>3 else ''}")
        model = model.to(device).eval()
        return model, VGGISH_SAMPLE_RATE

    raise ValueError(f"Unknown --model-type '{args.model_type}'")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_root: Path, split: str | None, n_samples: int | None,
               seed: int = 42) -> list[tuple[Path, int]]:
    """Return balanced (path, label) pairs from the CSV split."""
    samples: list[tuple[Path, int]] = []

    split_csv = None
    if split:
        for c in data_root.glob("*.csv"):
            if c.stem == split:
                split_csv = c
                break

    if split_csv and split_csv.exists():
        with open(split_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = data_root / row["filepath"]
                if not fp.exists():
                    for ext in (".wav", ".mp3", ".flac", ".ogg"):
                        alt = fp.with_suffix(ext)
                        if alt.exists():
                            fp = alt
                            break
                if fp.exists():
                    samples.append((fp, int(row["target"])))
    else:
        for label, subdir in ((0, "real_songs"), (1, "fake_songs")):
            folder = data_root / subdir
            if folder.is_dir():
                for ext in (".wav", ".mp3", ".flac"):
                    for p in sorted(folder.glob(f"*{ext}")):
                        samples.append((p, label))

    if n_samples is not None:
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


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_batch(model: torch.nn.Module, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model on [B, T]. Returns (preds [B], prob_fake [B])."""
    with torch.no_grad():
        logits = model(waveforms)
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs >= 0.5).long()
    else:
        probs = logits.softmax(dim=1)[:, 1]
        preds = logits.argmax(dim=1)
    return preds, probs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {
        "accuracy":         round(float(acc),  4),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1":               round(float(f1),   4),
        "auc":              round(float(auc),  4),
        "confusion_matrix": cm,
        "n_real":           int((labels == 0).sum()),
        "n_fake":           int((labels == 1).sum()),
    }


def _print_metrics(tag: str, m: dict) -> None:
    cm = m["confusion_matrix"]
    print(f"\n── {tag} ({'n=' + str(m['n_real'] + m['n_fake'])})")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}  (fake class)")
    print(f"  Recall    : {m['recall']:.4f}  (fake class)")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  AUC       : {m['auc']:.4f}")
    print(f"  Confusion  (rows=true, cols=pred):")
    print(f"             real   fake")
    print(f"    real  {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"    fake  {cm[1][0]:6d} {cm[1][1]:6d}")


# ── TensorBoard helpers ───────────────────────────────────────────────────────

def _make_spectrogram_image(wav: torch.Tensor, sample_rate: int,
                             max_seconds: float = 10.0,
                             cmap: str = "inferno") -> torch.Tensor:
    """Return [3, n_mels, T] RGB float tensor in [0, 1] suitable for add_image."""
    max_samples = int(max_seconds * sample_rate)
    y = wav.cpu().numpy()
    if y.shape[-1] > max_samples:
        y = y[..., :max_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    lo, hi = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - lo) / (hi - lo + 1e-8)
    mel_norm = mel_norm[::-1]

    colormap = plt.get_cmap(cmap)
    rgb = colormap(mel_norm)[:, :, :3]
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    p.add_argument("--config", default=None, metavar="YAML",
                   help="YAML config file; CLI flags override values defined here")

    # Model
    p.add_argument("--model-type", required=False, choices=["sonics", "ast", "vggish"])
    p.add_argument("--model-id",   default="awsaf49/sonics-spectttra-gamma-120s",
                   help="HuggingFace repo ID (sonics only)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Lightning .ckpt file (ast / vggish)")

    # Data
    p.add_argument("--data-root",    type=Path, required=False)
    p.add_argument("--split",        default="test")
    p.add_argument("--n-samples",    type=int, default=None,
                   help="Cap inference to N balanced samples (None = all)")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate",  type=int, default=16_000,
                   help="Overridden for ast/vggish automatically")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--seed",         type=int, default=42)

    # Output
    p.add_argument("--log-dir", type=Path, default=Path("runs/predict"))
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")

    # set_defaults must come after all add_argument calls so it can update action.default
    if _pre_args.config:
        from audio_xai.hparams import predict_defaults
        p.set_defaults(**predict_defaults(_pre_args.config))

    args = p.parse_args()

    if args.model_type is None:
        p.error("--model-type is required (or set model.name in your --config YAML)")
    if args.data_root is None:
        p.error("--data-root is required (or set data.root in your --config YAML)")

    print(f"Device : {args.device}")
    print(f"Model  : {args.model_type}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, sample_rate = load_model(args)
    clip_len = int(args.clip_seconds * sample_rate)
    print(f"Sample rate: {sample_rate} Hz  |  clip: {args.clip_seconds}s ({clip_len} samples)")

    # ── 2. Load split ─────────────────────────────────────────────────────────
    samples = load_split(args.data_root, args.split or None, args.n_samples, seed=args.seed)
    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    print(f"\nSplit '{args.split}': {len(samples)} files  ({n_real} real, {n_fake} fake)")
    if not samples:
        print("No files found — check --data-root and --split.")
        return

    # ── 3. Batch inference ────────────────────────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []
    errors = 0
    t0 = time.time()

    with tqdm(total=len(samples), desc="Inference", unit="clip",
              dynamic_ncols=True) as pbar:
        for batch_start in range(0, len(samples), args.batch_size):
            batch = samples[batch_start : batch_start + args.batch_size]
            waveforms, labels = [], []
            for path, label in batch:
                try:
                    waveforms.append(load_waveform(path, sample_rate, clip_len))
                    labels.append(label)
                except Exception as e:
                    tqdm.write(f"  [warn] skip {path.name}: {e}")
                    errors += 1
            if not waveforms:
                pbar.update(len(batch))
                continue
            preds, probs = predict_batch(model, torch.stack(waveforms).to(args.device))
            all_labels.extend(labels)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            pbar.update(len(batch))
            n_done = len(all_labels)
            if n_done:
                run_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n_done
                pbar.set_postfix(acc=f"{run_acc:.3f}", errors=errors)

    elapsed = time.time() - t0
    print(f"\nInference done: {len(all_labels)} clips in {elapsed:.1f}s  "
          f"({len(all_labels)/elapsed:.1f} clips/s, {errors} skipped)")

    labels_np = np.array(all_labels)
    preds_np  = np.array(all_preds)
    probs_np  = np.array(all_probs)
    orig_metrics = compute_metrics(labels_np, preds_np, probs_np)
    _print_metrics("Predictions", orig_metrics)

    # ── 4. Save + TensorBoard ─────────────────────────────────────────────────
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else args.model_id.replace("/", "_")
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{args.model_type}_{ckpt_tag}_{args.split}_{ts}"
    log_path = args.log_dir / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "run": {
            "model_type":   args.model_type,
            "model_id":     args.model_id if args.model_type == "sonics" else None,
            "checkpoint":   str(args.checkpoint) if args.checkpoint else None,
            "split":        args.split,
            "n_samples":    len(all_labels),
            "clip_seconds": args.clip_seconds,
            "sample_rate":  sample_rate,
        },
        "metrics": orig_metrics,
        "per_sample": [
            {"file": str(samples[i][0].name), "label": int(all_labels[i]),
             "pred": int(all_preds[i]), "prob_fake": round(float(all_probs[i]), 6)}
            for i in range(len(all_labels))
        ],
    }
    out_json = log_path / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nResults → {out_json}")

    writer = SummaryWriter(log_dir=str(log_path))

    def _metrics_md_table(metrics: dict, title: str) -> str:
        rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items() if isinstance(v, float))
        return f"**{title}**\n\n| Metric | Value |\n|--------|-------|\n{rows}"

    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        writer.add_scalar(f"predict/{k}", orig_metrics[k], 0)
    writer.add_text("metrics/predict", _metrics_md_table(orig_metrics, "Prediction (full test set)"), 0)
    writer.add_histogram("predict/score_real_class", probs_np[labels_np == 0])
    writer.add_histogram("predict/score_fake_class", probs_np[labels_np == 1])

    real_spec_samples = [(p, l) for p, l in samples if l == 0][:3]
    fake_spec_samples = [(p, l) for p, l in samples if l == 1][:3]
    for i, (path, _) in enumerate(real_spec_samples):
        try:
            wav = load_waveform(path, sample_rate, clip_len)
            writer.add_image(f"spectrogram/real/sample_{i}",
                             _make_spectrogram_image(wav, sample_rate), 0)
        except Exception:
            pass
    for i, (path, _) in enumerate(fake_spec_samples):
        try:
            wav = load_waveform(path, sample_rate, clip_len)
            writer.add_image(f"spectrogram/fake/sample_{i}",
                             _make_spectrogram_image(wav, sample_rate), 0)
        except Exception:
            pass

    writer.close()
    print(f"TensorBoard → {log_path}")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()