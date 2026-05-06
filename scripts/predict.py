"""Real/fake audio prediction + perceptual XAI attack for AST, VGGish, and Sonics.

Phase 1 — Predict: batch inference over the full split with accuracy/F1/AUC.
Phase 2 — Attack (optional, --attack): run the perceptual XAI attack on a small
  balanced batch, then compare original vs adversarial classification metrics and
  Grad-CAM cosine similarity in TensorBoard.

Usage examples:
    # Prediction only
    python scripts/predict.py --model-type sonics \\
        --model-id awsaf49/sonics-spectttra-gamma-120s \\
        --clip-seconds 120.0 --data-root audio_xai/data/external

    python scripts/predict.py --model-type ast \\
        --checkpoint runs/ast/version_2/checkpoints/epoch=2-step=1500.ckpt \\
        --data-root audio_xai/data/external

    # Prediction + attack
    python scripts/predict.py --model-type vggish \\
        --checkpoint runs/vggish/version_3/checkpoints/epoch=2-step=750.ckpt \\
        --data-root audio_xai/data/external \\
        --attack --n-attack-samples 10 --n-steps 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.tensorboard import SummaryWriter

from audio_xai.attacks.perceptual_xai_attack import (
    AttackConfig, AttackResult, perceptual_xai_attack, topk_overlap,
)
from audio_xai.fetching_and_metrics.peaq_implementation import peaq_like
from audio_xai.fetching_and_metrics.preprocessing_metrics import (
    PESQ_SR, compute_pesq, compute_stoi,
)
from audio_xai.models.ast_binary import ASTBinary, AST_SAMPLE_RATE
from audio_xai.models.base import AudioClassifier
from audio_xai.models.vggish_binary import VGGishBinary, VGGISH_SAMPLE_RATE
from audio_xai.xai.gradcam import GradCAMBase, make_gradcam


# ── Sonics wrapper (for Grad-CAM / attack) ────────────────────────────────────

class _SonicsWrapper(AudioClassifier):
    """Adapts HFAudioClassifier to AudioClassifier so make_gradcam / attack work."""

    def __init__(self, sonics_model) -> None:
        nn.Module.__init__(self)
        self._m = sonics_model

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self._m.ft_extractor(waveform)
        spec = spec.unsqueeze(1)
        return F.interpolate(
            spec, size=tuple(self._m.input_shape), mode="bilinear", align_corners=False
        )

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self._m.encoder(features)
        embeds = tokens.mean(dim=1)
        return self._m.classifier(embeds)

    @property
    def target_layer(self) -> nn.Module:
        return self._m.encoder.transformer.blocks[-1]


class _SonicsGradCAM(GradCAMBase):
    def _build_heatmap(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        weights = gradients.mean(dim=2, keepdim=True)
        cam = (weights * activations).sum(dim=2)
        return F.relu(cam)


# ── Model loading ─────────────────────────────────────────────────────────────

def _strip_lightning_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_"):
            continue
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def load_model(args) -> tuple[torch.nn.Module, torch.nn.Module, int]:
    """Return (inference_model, attack_model, sample_rate).

    inference_model  — used for plain predict_batch (may be HFAudioClassifier)
    attack_model     — AudioClassifier subclass usable by the attack / Grad-CAM
    Both point to the same underlying weights.
    """
    device = args.device

    if args.model_type == "sonics":
        from sonics import HFAudioClassifier
        print(f"Loading Sonics model: {args.model_id}")
        raw = HFAudioClassifier.from_pretrained(args.model_id, map_location=device)
        raw = raw.to(device).eval()
        print(f"  input_shape={raw.input_shape}  n_classes={raw.num_classes}")
        wrapped = _SonicsWrapper(raw).to(device).eval()
        return raw, wrapped, args.sample_rate

    if args.model_type == "ast":
        print("Loading ASTBinary…")
        model = ASTBinary(pretrained=True)
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
        return model, model, AST_SAMPLE_RATE

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
        return model, model, VGGISH_SAMPLE_RATE

    raise ValueError(f"Unknown --model-type '{args.model_type}'")


def _make_gradcam(model_type: str, attack_model: AudioClassifier) -> GradCAMBase:
    if model_type == "sonics":
        return _SonicsGradCAM(attack_model)
    return make_gradcam(attack_model)   # handles ast / vggish


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
                    # CSV may list a different extension than what's on disk
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
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    if wav.shape[0] < clip_len:
        wav = torch.nn.functional.pad(wav, (0, clip_len - wav.shape[0]))
    else:
        wav = wav[:clip_len]
    return wav


def load_balanced_batch(
    samples: list[tuple[Path, int]],
    n: int,
    sample_rate: int,
    clip_len: int,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load n waveforms (n/2 real, n/2 fake) into a GPU tensor. Returns (x, labels)."""
    rng = random.Random(seed)
    real = [(p, l) for p, l in samples if l == 0]
    fake = [(p, l) for p, l in samples if l == 1]
    rng.shuffle(real); rng.shuffle(fake)
    chosen = real[: n // 2] + fake[: n // 2]
    rng.shuffle(chosen)

    waveforms, labels = [], []
    for path, label in chosen:
        try:
            waveforms.append(load_waveform(path, sample_rate, clip_len))
            labels.append(label)
        except Exception as e:
            print(f"  [warn] skip {path.name}: {e}")

    x = torch.stack(waveforms).to(device)
    gt = torch.tensor(labels, dtype=torch.long)
    return x, gt


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


# ── Attack micro-batching ─────────────────────────────────────────────────────

def _run_attack_chunked(
    model_type: str,
    attack_model,
    x_atk: torch.Tensor,
    cfg: "AttackConfig",
    micro_bs: int,
) -> "AttackResult":
    """Run the perceptual XAI attack in GPU micro-batches and concatenate results.

    Each sample has its own independent delta (the batch loss is just an average),
    so attacking in chunks of micro_bs is mathematically equivalent to the full
    batch — without the OOM. Grad-CAM hooks are re-registered for every chunk
    because perceptual_xai_attack removes them at the end.
    """
    n = x_atk.shape[0]
    chunks = [x_atk[i : i + micro_bs] for i in range(0, n, micro_bs)]

    parts_x_adv, parts_delta, parts_cam_orig, parts_cam_adv = [], [], [], []
    parts_cos_sim, parts_pred_pres = [], []
    history_chunks: list[list[dict]] = []

    for idx, chunk in enumerate(chunks):
        tqdm.write(f"  micro-batch {idx + 1}/{len(chunks)}  "
                   f"({chunk.shape[0]} samples)")
        gradcam = _make_gradcam(model_type, attack_model)
        result = perceptual_xai_attack(attack_model, chunk, cfg, gradcam=gradcam)
        parts_x_adv.append(result.x_adv)
        parts_delta.append(result.delta)
        parts_cam_orig.append(result.cam_original)
        parts_cam_adv.append(result.cam_adv)
        parts_cos_sim.append(result.cosine_similarity)
        parts_pred_pres.append(result.prediction_preserved)
        history_chunks.append(result.history)
        torch.cuda.empty_cache()

    # Average loss history across chunks so the loss curve reflects the whole run
    n_steps = max(len(h) for h in history_chunks)
    merged_history: list[dict] = []
    for s in range(n_steps):
        entries = [h[s] for h in history_chunks if s < len(h)]
        merged_history.append({
            k: sum(e[k] for e in entries) / len(entries)
            for k in entries[0]
        })

    return AttackResult(
        x_adv=torch.cat(parts_x_adv),
        delta=torch.cat(parts_delta),
        cam_original=torch.cat(parts_cam_orig),
        cam_adv=torch.cat(parts_cam_adv),
        cosine_similarity=torch.cat(parts_cos_sim),
        prediction_preserved=torch.cat(parts_pred_pres),
        history=merged_history,
    )


# ── TensorBoard helpers ───────────────────────────────────────────────────────

def _make_spectrogram_image(wav: torch.Tensor, sample_rate: int,
                             max_seconds: float = 10.0,
                             cmap: str = "inferno") -> torch.Tensor:
    """Return [3, n_mels, T] RGB float tensor in [0, 1] suitable for add_image."""
    max_samples = int(max_seconds * sample_rate)
    w = wav.cpu()
    if w.shape[-1] > max_samples:
        w = w[..., :max_samples]
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80
    )
    mel = transform(w)                          # [n_mels, T]
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel.unsqueeze(0), multiplier=10.0, amin=1e-5, db_multiplier=0.0, top_db=80.0
    )                                           # [1, n_mels, T]
    lo, hi = mel_db.min(), mel_db.max()
    mel_norm = ((mel_db - lo) / (hi - lo + 1e-8)).squeeze(0).flip(dims=[0])  # [n_mels, T]

    # Apply colormap → [H, W, 4] RGBA numpy, keep RGB, convert to [3, H, W] tensor
    colormap = plt.get_cmap(cmap)
    rgb = colormap(mel_norm.numpy())[:, :, :3]          # [H, W, 3]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()  # [3, H, W]


def _metrics_bar_figure(orig: dict, adv: dict, title: str = "") -> plt.Figure:
    """Grouped bar chart: original vs adversarial for accuracy, F1, AUC."""
    keys   = ["accuracy", "f1", "auc"]
    labels = ["Accuracy", "F1", "AUC"]
    x = np.arange(len(keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, [orig[k] for k in keys], w, label="Original",    color="#4C72B0")
    b2 = ax.bar(x + w / 2, [adv[k]  for k in keys], w, label="Adversarial", color="#DD8452")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title(title or "Original vs Adversarial Metrics")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    return fig


def _psychoacoustic_one(
    i: int,
    orig: torch.Tensor,
    adv: torch.Tensor,
    sample_rate: int,
) -> dict:
    """Compute PESQ, STOI, ViSQOL, PEAQ, Zimtohrli for one sample. CPU, thread-safe."""
    m: dict = {}

    if sample_rate != PESQ_SR:
        orig_16k = torchaudio.functional.resample(orig, sample_rate, PESQ_SR)
        adv_16k  = torchaudio.functional.resample(adv,  sample_rate, PESQ_SR)
    else:
        orig_16k, adv_16k = orig, adv
    orig_np = orig_16k.numpy()
    adv_np  = adv_16k.numpy()

    try:
        m["pesq"] = round(compute_pesq(orig_16k, adv_16k, sr=PESQ_SR), 6)
    except Exception as e:
        m["pesq"] = None
        print(f"  [warn] PESQ [{i}]: {e}")

    try:
        m["stoi"] = round(compute_stoi(orig_np, adv_np, sr=PESQ_SR), 6)
    except Exception as e:
        m["stoi"] = None
        print(f"  [warn] STOI [{i}]: {e}")

    try:
        from visqol import VisqolApi
        api = VisqolApi()
        api.create(mode="speech")
        result = api.measure_from_arrays(
            orig_np.astype(np.float64), adv_np.astype(np.float64), sample_rate=PESQ_SR,
        )
        m["visqol"] = round(float(result.moslqo), 6)
    except Exception as e:
        m["visqol"] = None
        print(f"  [warn] ViSQOL [{i}]: {e}")

    ref_path = deg_path = None
    try:
        fd_r, ref_path = tempfile.mkstemp(suffix=".wav")
        fd_d, deg_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_r); os.close(fd_d)
        orig_48k = torchaudio.functional.resample(orig, sample_rate, 48_000).unsqueeze(0)
        adv_48k  = torchaudio.functional.resample(adv,  sample_rate, 48_000).unsqueeze(0)
        torchaudio.save(ref_path, orig_48k, 48_000)
        torchaudio.save(deg_path, adv_48k,  48_000)
        m["peaq"] = round(peaq_like(ref_path, deg_path), 6)
    except Exception as e:
        m["peaq"] = None
        print(f"  [warn] PEAQ [{i}]: {e}")
    finally:
        for p in filter(None, [ref_path, deg_path]):
            try:
                os.unlink(p)
            except OSError:
                pass

    try:
        from zimtohrli import mos_from_signals
        orig_48k_np = torchaudio.functional.resample(orig, sample_rate, 48_000).numpy()
        adv_48k_np  = torchaudio.functional.resample(adv,  sample_rate, 48_000).numpy()
        m["zimtohrli"] = round(float(mos_from_signals(orig_48k_np, adv_48k_np)), 6)
    except Exception as e:
        m["zimtohrli"] = None
        print(f"  [warn] Zimtohrli [{i}]: {e}")

    return m


def _compute_psychoacoustic_metrics(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    sample_rate: int,
) -> list[dict]:
    """Compute psychoacoustic quality metrics for all original/adversarial pairs in parallel."""
    n = x_orig.shape[0]
    origs = [x_orig[i].cpu() for i in range(n)]
    advs  = [x_adv[i].cpu()  for i in range(n)]
    n_workers = min(n, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_psychoacoustic_one, i, origs[i], advs[i], sample_rate)
                   for i in range(n)]
        return [f.result() for f in futures]


def _psychoacoustic_bar_figure(audio_metrics: list[dict], title: str = "") -> plt.Figure:
    """Horizontal bar chart of mean perceptual audio quality metrics (orig → adv)."""
    METRICS = [
        ("pesq",      "PESQ (1–4.5 ↑)",        "#4C72B0"),
        ("stoi",      "STOI (0–1 ↑)",           "#55A868"),
        ("visqol",    "ViSQOL MOS (1–5 ↑)",     "#C44E52"),
        ("peaq",      "PEAQ (−3.5–0 ↑)",        "#8172B3"),
        ("zimtohrli", "Zimtohrli MOS (1–5 ↑)",  "#CCB974"),
    ]

    rows = []
    for key, label, color in METRICS:
        vals = [m[key] for m in audio_metrics if m.get(key) is not None]
        if vals:
            rows.append((label, float(np.mean(vals)), color))

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No perceptual metrics available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.9 * len(rows) + 1.5)))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, height=0.55, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean score")
    ax.set_title(title or "Perceptual Audio Quality (original → adversarial)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        offset = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
        ha = "left" if val >= 0 else "right"
        x_pos = bar.get_width() + (offset if val >= 0 else -offset)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=10, fontweight="bold")

    ax.margins(x=0.18)
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model
    p.add_argument("--model-type", required=True, choices=["sonics", "ast", "vggish"])
    p.add_argument("--model-id",   default="awsaf49/sonics-spectttra-gamma-120s",
                   help="HuggingFace repo ID (sonics only)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Lightning .ckpt file (ast / vggish)")

    # Data
    p.add_argument("--data-root",    type=Path, required=True)
    p.add_argument("--split",        default="test")
    p.add_argument("--n-samples",    type=int, default=None,
                   help="Cap inference to N balanced samples (None = all)")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate",  type=int, default=16_000,
                   help="Overridden for ast/vggish automatically")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--seed",         type=int, default=42)

    # Attack (optional)
    p.add_argument("--attack",           action="store_true",
                   help="Run perceptual XAI attack after regular prediction")
    p.add_argument("--n-attack-samples", type=int, default=10,
                   help="Number of samples to attack (balanced real/fake)")
    p.add_argument("--attack-micro-batch", type=int, default=None,
                   help="GPU micro-batch size for the attack. Use a small value "
                        "(e.g. 4) for large transformers (AST/Sonics) to avoid OOM. "
                        "Defaults to --n-attack-samples (single pass).")
    p.add_argument("--n-steps",          type=int, default=50,
                   help="Adam optimisation steps for the attack")
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--lambda-aud",       type=float, default=1.0)
    p.add_argument("--lambda-pred",      type=float, default=100.0)

    # Output
    p.add_argument("--log-dir", type=Path, default=Path("runs/predict"))
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Device : {args.device}")
    print(f"Model  : {args.model_type}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    infer_model, attack_model, sample_rate = load_model(args)
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
    n_batches = (len(samples) + args.batch_size - 1) // args.batch_size

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
            preds, probs = predict_batch(infer_model, torch.stack(waveforms).to(args.device))
            all_labels.extend(labels)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            pbar.update(len(batch))
            # Running accuracy shown inline
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
    _print_metrics("Original predictions", orig_metrics)

    # ── 4. Attack (optional) ──────────────────────────────────────────────────
    adv_metrics: dict | None = None
    orig_atk_metrics: dict | None = None
    attack_result: AttackResult | None = None
    gt_attack: torch.Tensor | None = None
    x_atk: torch.Tensor | None = None

    if args.attack:
        print(f"\n── Perceptual XAI attack  "
              f"({args.n_attack_samples} samples, {args.n_steps} steps) ───────")

        x_atk, gt_attack = load_balanced_batch(
            samples, args.n_attack_samples, sample_rate, clip_len, args.device, seed=args.seed
        )
        print(f"Attack batch: {list(x_atk.shape)}  "
              f"(real={(gt_attack==0).sum().item()}, fake={(gt_attack==1).sum().item()})")

        # Baseline on the same attack batch (for a fair comparison)
        preds_orig_atk, probs_orig_atk = predict_batch(infer_model, x_atk)
        orig_atk_metrics = compute_metrics(
            gt_attack.cpu().numpy(),
            preds_orig_atk.cpu().numpy(),
            probs_orig_atk.cpu().numpy(),
        )
        _print_metrics("Original (attack batch)", orig_atk_metrics)

        cfg = AttackConfig(
            n_steps=args.n_steps,
            lr=args.lr,
            lambda_audibility=args.lambda_aud,
            lambda_pred=args.lambda_pred,
            log_every=1,
            sample_rate=sample_rate,
        )

        micro_bs = args.attack_micro_batch or args.n_attack_samples
        n_chunks = (args.n_attack_samples + micro_bs - 1) // micro_bs
        print(f"  micro-batch size: {micro_bs}  ({n_chunks} chunk(s))")

        t_atk = time.time()
        if micro_bs >= args.n_attack_samples:
            gradcam = _make_gradcam(args.model_type, attack_model)
            attack_result = perceptual_xai_attack(attack_model, x_atk, cfg, gradcam=gradcam)
        else:
            attack_result = _run_attack_chunked(
                args.model_type, attack_model, x_atk, cfg, micro_bs
            )
        print(f"Attack done in {time.time()-t_atk:.1f}s")

        preds_adv, probs_adv = predict_batch(infer_model, attack_result.x_adv)
        adv_metrics = compute_metrics(
            gt_attack.cpu().numpy(),
            preds_adv.cpu().numpy(),
            probs_adv.cpu().numpy(),
        )
        _print_metrics("Adversarial predictions", adv_metrics)

        cos  = attack_result.cosine_similarity
        over = topk_overlap(attack_result.cam_original, attack_result.cam_adv, k_frac=0.1)
        print(f"\n  CAM cosine sim  : mean={cos.mean():.4f}  (lower = more explanation change)")
        print(f"  Top-10% overlap : mean={over.mean():.4f}  (lower = more disagreement)")
        print(f"  δ L∞            : {attack_result.delta.abs().max():.5f}")
        print(f"  Pred preserved  : "
              f"{attack_result.prediction_preserved.sum().item()}/{args.n_attack_samples}")

        print(f"\n── Δ (adversarial − original, same batch) ──────────────────────────")
        for k in ("accuracy", "f1", "auc"):
            orig_v = orig_atk_metrics[k]
            adv_v  = adv_metrics[k]
            print(f"  Δ{k:10s}: {adv_v - orig_v:+.4f}  ({orig_v:.4f} → {adv_v:.4f})")

    # ── 5. Save + TensorBoard ─────────────────────────────────────────────────
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else args.model_id.replace("/", "_")
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{args.model_type}_{ckpt_tag}_{args.split}_{ts}"
    log_path = args.log_dir / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "run": {
            "model_type":       args.model_type,
            "model_id":         args.model_id if args.model_type == "sonics" else None,
            "checkpoint":       str(args.checkpoint) if args.checkpoint else None,
            "split":            args.split,
            "n_samples":        len(all_labels),
            "clip_seconds":     args.clip_seconds,
            "sample_rate":      sample_rate,
            "attack":           args.attack,
            "n_attack_samples": args.n_attack_samples if args.attack else None,
            "n_steps":          args.n_steps if args.attack else None,
        },
        "original_metrics":         orig_metrics,
        "original_metrics_atk_batch": orig_atk_metrics,
        "adversarial_metrics":      adv_metrics,
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

    # ── Custom scalars layout (groups tags onto shared charts, no sub-runs) ───
    # Must be called before writing any scalars.
    tb_layout: dict = {
        "1 — Prediction": {
            "Accuracy / F1 / AUC": ["Multiline", [
                "predict/accuracy", "predict/f1", "predict/auc",
            ]],
        },
    }
    if args.attack:
        tb_layout["2 — Attack losses"] = {
            "All losses":       ["Multiline", ["loss/total", "loss/explain",
                                               "loss/audibility", "loss/pred"]],
            "Cosine similarity": ["Multiline", ["loss/cos_sim"]],
        }
        tb_layout["3 — Attack result"] = {
            "Accuracy":  ["Multiline", ["metrics/original_accuracy",  "metrics/adv_accuracy"]],
            "F1 score":  ["Multiline", ["metrics/original_f1",        "metrics/adv_f1"]],
            "AUC":       ["Multiline", ["metrics/original_auc",       "metrics/adv_auc"]],
            "Δ metrics": ["Multiline", ["metrics/delta_accuracy",     "metrics/delta_f1",
                                        "metrics/delta_auc"]],
        }
    writer.add_custom_scalars(tb_layout)

    # ── Prediction metrics (full test set) ────────────────────────────────────
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        writer.add_scalar(f"predict/{k}", orig_metrics[k], 0)
    writer.add_histogram("predict/score_real_class", probs_np[labels_np == 0])
    writer.add_histogram("predict/score_fake_class", probs_np[labels_np == 1])

    # ── Spectrograms: a few test samples ─────────────────────────────────────
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

    # ── Attack section ────────────────────────────────────────────────────────
    if adv_metrics is not None and orig_atk_metrics is not None \
            and attack_result is not None and x_atk is not None:

        # Bar-chart figure: original vs adversarial (same batch)
        fig = _metrics_bar_figure(
            orig_atk_metrics, adv_metrics,
            title=f"{args.model_type} — Original vs Adversarial ({args.n_attack_samples} samples)"
        )
        writer.add_figure("attack/metrics_comparison", fig, 0)
        plt.close(fig)

        # Flat add_scalar calls — NO add_scalars (which creates sub-run dirs)
        for k in ("accuracy", "f1", "auc"):
            writer.add_scalar(f"metrics/original_{k}", orig_atk_metrics[k], 0)
            writer.add_scalar(f"metrics/adv_{k}",      adv_metrics[k],      0)
            writer.add_scalar(f"metrics/delta_{k}",
                              adv_metrics[k] - orig_atk_metrics[k], 0)

        # Loss curves — all under loss/ prefix, same event file
        for entry in attack_result.history:
            s = entry["step"]
            writer.add_scalar("loss/total",      entry["loss"],             s)
            writer.add_scalar("loss/explain",    entry["loss_explain"],     s)
            writer.add_scalar("loss/audibility", entry["loss_audibility"],  s)
            writer.add_scalar("loss/pred",       entry["loss_pred"],        s)
            writer.add_scalar("loss/cos_sim",    entry["cos_sim"],          s)

        # Final attack summary
        cos  = attack_result.cosine_similarity
        over = topk_overlap(attack_result.cam_original, attack_result.cam_adv, k_frac=0.1)
        writer.add_scalar("attack/mean_cos_sim",        cos.mean().item(),  0)
        writer.add_scalar("attack/top10_overlap",       over.mean().item(), 0)
        writer.add_scalar("attack/delta_linf",          attack_result.delta.abs().max().item(), 0)
        writer.add_scalar("attack/pred_preserved_frac",
                          attack_result.prediction_preserved.float().mean().item(), 0)

        # Perceptual audio quality metrics (PESQ / STOI / ViSQOL / PEAQ / Zimtohrli)
        adv_cpu   = attack_result.x_adv.cpu()
        x_atk_cpu = x_atk.cpu()
        print("\n── Perceptual audio quality metrics (orig → adv) ───────────────────")
        audio_quality = _compute_psychoacoustic_metrics(x_atk_cpu, adv_cpu, sample_rate)
        for key in ("pesq", "stoi", "visqol", "peaq", "zimtohrli"):
            vals = [m[key] for m in audio_quality if m.get(key) is not None]
            if vals:
                print(f"  {key.upper():10s}: mean={np.mean(vals):.4f}  "
                      f"values={[f'{v:.3f}' for v in vals]}")
                writer.add_scalar(f"perceptual/mean_{key}", float(np.mean(vals)), 0)

        fig = _psychoacoustic_bar_figure(
            audio_quality,
            title=f"{args.model_type} — Perceptual Quality ({args.n_attack_samples} samples)",
        )
        writer.add_figure("attack/metrics_comparison_psychoacoustic", fig, 0)
        plt.close(fig)

        # Spectrograms + audio in TensorBoard
        n_show = min(3, x_atk.shape[0])
        delta_cpu = attack_result.delta.cpu()
        for i in range(n_show):
            lbl = "real" if (gt_attack is not None and gt_attack[i].item() == 0) else "fake"
            writer.add_image(f"spectrogram/attack_original/{lbl}_{i}",
                             _make_spectrogram_image(x_atk_cpu[i], sample_rate), 0)
            writer.add_image(f"spectrogram/attack_adversarial/{lbl}_{i}",
                             _make_spectrogram_image(adv_cpu[i], sample_rate), 0)
            writer.add_image(f"spectrogram/attack_delta/{lbl}_{i}",
                             _make_spectrogram_image(delta_cpu[i], sample_rate), 0)
            writer.add_audio(f"audio/original/{lbl}_{i}",
                             x_atk_cpu[i].unsqueeze(0), sample_rate=sample_rate)
            writer.add_audio(f"audio/adversarial/{lbl}_{i}",
                             adv_cpu[i].unsqueeze(0), sample_rate=sample_rate)

        # ── Save artifacts to disk (mirrors sonics_attack_test structure) ──────
        audio_dir   = log_path / "audio"
        heatmap_dir = log_path / "heatmaps"
        audio_dir.mkdir(exist_ok=True)
        heatmap_dir.mkdir(exist_ok=True)

        n_atk = x_atk.shape[0]
        print(f"\nSaving attack artifacts → {log_path}")
        for i in range(n_atk):
            torchaudio.save(str(audio_dir / f"sample_{i:02d}_original.wav"),
                            x_atk_cpu[i].unsqueeze(0), sample_rate)
            torchaudio.save(str(audio_dir / f"sample_{i:02d}_adversarial.wav"),
                            adv_cpu[i].unsqueeze(0), sample_rate)
            torchaudio.save(str(audio_dir / f"sample_{i:02d}_delta.wav"),
                            delta_cpu[i].unsqueeze(0), sample_rate)
            np.save(str(heatmap_dir / f"sample_{i:02d}_original.npy"),
                    attack_result.cam_original[i].cpu().numpy())
            np.save(str(heatmap_dir / f"sample_{i:02d}_adversarial.npy"),
                    attack_result.cam_adv[i].cpu().numpy())

        (log_path / "loss_history.json").write_text(
            json.dumps(attack_result.history, indent=2)
        )

        print(f"  audio/        → {n_atk * 3} wav files")
        print(f"  heatmaps/     → {n_atk * 2} npy files")
        print(f"  loss_history.json")

    writer.close()
    print(f"TensorBoard → {log_path}")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
