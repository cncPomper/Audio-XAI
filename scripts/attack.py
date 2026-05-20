"""Perceptual XAI attack for AST, VGGish, and Sonics.

Runs the perceptual XAI attack on a small balanced batch, then compares
original vs adversarial classification metrics and Grad-CAM cosine
similarity in TensorBoard.

Usage examples:
    python scripts/attack.py --model-type sonics \\
        --model-id awsaf49/sonics-spectttra-gamma-120s \\
        --clip-seconds 120.0 --data-root audio_xai/data/external \\
        --n-attack-samples 10 --n-steps 50

    python scripts/attack.py --model-type ast \\
        --checkpoint runs/ast/version_2/checkpoints/epoch=2-step=1500.ckpt \\
        --data-root audio_xai/data/external \\
        --n-attack-samples 10 --n-steps 50

    python scripts/attack.py --model-type vggish \\
        --checkpoint runs/vggish/version_3/checkpoints/epoch=2-step=750.ckpt \\
        --data-root audio_xai/data/external \\
        --n-attack-samples 10 --n-steps 50

    # Using a YAML config (reads [model], [predict.split], [attack] sections)
    python scripts/attack.py --config config/predict_ast.yaml \\
        --data-root audio_xai/data/external
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

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
    return make_gradcam(attack_model)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_root: Path, split: str | None,
               seed: int = 42) -> list[tuple[Path, int]]:
    """Return (path, label) pairs from the CSV split."""
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

    return samples


def load_waveform(path: Path, sample_rate: int, clip_len: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    if wav.shape[0] < clip_len:
        wav = torch.nn.functional.pad(wav, (0, clip_len - wav.shape[0]))
    else:
        wav = wav[:clip_len]
    return wav


def load_full_waveform(path: Path, sample_rate: int) -> torch.Tensor:
    """Load the entire audio file without any length clipping. Returns 1-D tensor [T]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)


def _select_balanced_batch(
    samples: list[tuple[Path, int]],
    n: int,
    seed: int = 0,
) -> tuple[list[Path], list[int]]:
    """Select n balanced paths+labels (n/2 real, n/2 fake) without loading audio."""
    rng = random.Random(seed)
    real = [(p, l) for p, l in samples if l == 0]
    fake = [(p, l) for p, l in samples if l == 1]
    rng.shuffle(real)
    rng.shuffle(fake)
    chosen = real[: n // 2] + fake[: n // 2]
    rng.shuffle(chosen)
    return [p for p, _ in chosen], [l for _, l in chosen]


def load_balanced_batch(
    samples: list[tuple[Path, int]],
    n: int,
    sample_rate: int,
    clip_len: int,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[Path]]:
    """Load n waveforms (n/2 real, n/2 fake) into a GPU tensor. Returns (x, labels, paths)."""
    rng = random.Random(seed)
    real = [(p, l) for p, l in samples if l == 0]
    fake = [(p, l) for p, l in samples if l == 1]
    rng.shuffle(real); rng.shuffle(fake)
    chosen = real[: n // 2] + fake[: n // 2]
    rng.shuffle(chosen)

    waveforms, labels, paths = [], [], []
    for path, label in chosen:
        try:
            waveforms.append(load_waveform(path, sample_rate, clip_len))
            labels.append(label)
            paths.append(path)
        except Exception as e:
            print(f"  [warn] skip {path.name}: {e}")

    x = torch.stack(waveforms).to(device)
    gt = torch.tensor(labels, dtype=torch.long)
    return x, gt, paths


# ── Dataset ───────────────────────────────────────────────────────────────────

class AudioPathDataset(Dataset):
    """Yields (path_str, label) pairs without loading audio.

    Audio loading happens inside AttackModule.predict_step so each sample is
    processed and freed independently, preventing cross-sample GPU memory growth.
    """

    def __init__(self, paths: list[Path], labels: list[int]) -> None:
        self.paths  = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return str(self.paths[idx]), self.labels[idx]


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
    device: str | None = None,
) -> "AttackResult":
    """Run the perceptual XAI attack in GPU micro-batches and concatenate results.

    Each sample has its own independent delta (the batch loss is just an average),
    so attacking in chunks of micro_bs is mathematically equivalent to the full
    batch — without the OOM. Grad-CAM hooks are re-registered for every chunk
    because perceptual_xai_attack removes them at the end.

    If `device` is provided, each chunk is moved there just before attacking and
    freed immediately after — x_atk can safely stay on CPU.
    """
    n = x_atk.shape[0]
    chunks = [x_atk[i : i + micro_bs] for i in range(0, n, micro_bs)]

    parts_x_adv, parts_delta, parts_cam_orig, parts_cam_adv = [], [], [], []
    parts_cos_sim, parts_pred_pres = [], []
    history_chunks: list[list[dict]] = []

    for idx, chunk in enumerate(chunks):
        tqdm.write(f"  micro-batch {idx + 1}/{len(chunks)}  "
                   f"({chunk.shape[0]} samples)")
        if device is not None:
            chunk = chunk.to(device)
        gradcam = _make_gradcam(model_type, attack_model)
        result = perceptual_xai_attack(attack_model, chunk, cfg, gradcam=gradcam)
        # .detach() BEFORE .cpu(): with torch.enable_grad() active, calling
        # .cpu() on a requires_grad tensor creates a CopyBackwards grad_fn that
        # holds a reference to the GPU tensor, preventing it from being freed
        # even after `del result`. Detaching severs the autograd link first.
        parts_x_adv.append(result.x_adv.detach().cpu())
        parts_delta.append(result.delta.detach().cpu())
        parts_cam_orig.append(result.cam_original.detach().cpu())
        parts_cam_adv.append(result.cam_adv.detach().cpu())
        parts_cos_sim.append(result.cosine_similarity.detach().cpu())
        parts_pred_pres.append(result.prediction_preserved.detach().cpu())
        history_chunks.append(result.history)
        del result, gradcam, chunk
        # Free model parameter .grad buffers populated by the second-order
        # GradCAM backward; set_to_none=True releases the storage entirely
        # rather than zeroing the existing buffer.
        attack_model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

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


# ── Sliding-window helpers ────────────────────────────────────────────────────

def _split_into_windows(
    x: torch.Tensor,
    clip_len: int,
    hop_len: int,
) -> tuple[torch.Tensor, int]:
    """Split a 1-D waveform [T] into [N, clip_len] windows with `hop_len` stride.

    The last window is zero-padded when the audio does not fill it completely.
    Returns (windows [N, clip_len], original_length T).
    """
    T = x.shape[0]
    windows = []
    for s in range(0, T, hop_len):
        chunk = x[s : s + clip_len]
        if chunk.shape[0] < clip_len:
            chunk = torch.nn.functional.pad(chunk, (0, clip_len - chunk.shape[0]))
        windows.append(chunk)
    return torch.stack(windows), T


def _stitch_delta(
    deltas: torch.Tensor,
    clip_len: int,
    hop_len: int,
    original_len: int,
) -> torch.Tensor:
    """Reconstruct a full-length delta from per-window deltas [N, clip_len].

    Uses overlap-add with a Hann window when windows overlap (hop_len < clip_len)
    for smooth transitions. Falls back to a rectangular window (simple tiling)
    when there is no overlap.
    """
    N = deltas.shape[0]
    buf = torch.zeros(original_len + clip_len, dtype=deltas.dtype)
    wgt = torch.zeros_like(buf)

    win = (
        torch.hann_window(clip_len, periodic=False)
        if hop_len < clip_len
        else torch.ones(clip_len, dtype=deltas.dtype)
    )

    for i in range(N):
        s = i * hop_len
        buf[s : s + clip_len] += win * deltas[i].cpu()
        wgt[s : s + clip_len] += win

    wgt.clamp_(min=1e-8)
    return (buf / wgt)[:original_len]


def _attack_full_audio_sample(
    model_type: str,
    attack_model,
    x_full: torch.Tensor,
    cfg: "AttackConfig",
    clip_len: int,
    hop_len: int,
    micro_bs: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, "AttackResult"]:
    """Attack a single full-length waveform via a sliding window.

    Splits x_full [T] (CPU) into overlapping clip_len windows, attacks every
    window in micro-batches of `micro_bs`, then stitches the per-window deltas
    back into a full-length delta via overlap-add.

    Args:
        x_full:   1-D CPU float32 tensor [T].
        clip_len: Window size in samples (clip_seconds × sample_rate).
        hop_len:  Hop size in samples.  Equal to clip_len → no overlap.
                  Smaller → overlapping windows stitched with a Hann window.
        micro_bs: Windows processed on GPU simultaneously (1 is safest for AST).

    Returns:
        x_adv_full [T], delta_full [T], window_result (AttackResult with N rows).
    """
    windows, T = _split_into_windows(x_full, clip_len, hop_len)
    n_win = windows.shape[0]
    overlap_pct = (1.0 - hop_len / clip_len) * 100.0
    print(f"  sliding-window: {T / cfg.sample_rate:.1f}s → {n_win} windows  "
          f"(clip={clip_len / cfg.sample_rate:.1f}s, "
          f"hop={hop_len / cfg.sample_rate:.1f}s, overlap={overlap_pct:.0f}%)")

    window_result = _run_attack_chunked(
        model_type, attack_model, windows, cfg, micro_bs, device=device,
    )

    delta_full = _stitch_delta(window_result.delta, clip_len, hop_len, T)
    x_adv_full = (x_full.cpu() + delta_full).clamp(-1.0, 1.0)
    return x_adv_full, delta_full, window_result


# ── TensorBoard helpers ───────────────────────────────────────────────────────

def _make_spectrogram_image(wav: torch.Tensor, sample_rate: int,
                             max_seconds: float | None = 10.0,
                             cmap: str = "inferno") -> torch.Tensor:
    """Return [3, n_mels, T] RGB float tensor in [0, 1] suitable for add_image."""
    y = wav.cpu().numpy()
    if max_seconds is not None:
        max_samples = int(max_seconds * sample_rate)
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


def _metrics_bar_figure(
    orig: dict,
    adv: dict,
    title: str = "",
    xai_summary: dict | None = None,
) -> plt.Figure:
    keys   = ["accuracy", "f1", "auc"]
    labels = ["Accuracy", "F1", "AUC"]
    x = np.arange(len(keys))
    w = 0.35

    ncols = 2 if xai_summary else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    ax = axes[0] if xai_summary else axes

    b1 = ax.bar(x - w / 2, [orig[k] for k in keys], w, label="Original",    color="#4C72B0")
    b2 = ax.bar(x + w / 2, [adv[k]  for k in keys], w, label="Adversarial", color="#DD8452")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title(title or "Classification metrics\n(flat = predictions preserved by design)")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)

    if xai_summary:
        ax2 = axes[1]
        xai_keys = [
            ("pred_preserved_frac",    "Pred\npreserved ↑", "#55A868"),
            ("1 - mean_cos_sim",       "Explanation\ndivergence ↑", "#C44E52"),
            ("1 - top10_overlap",      "Top-10%\nnon-overlap ↑", "#8172B2"),
        ]
        xai_vals, xai_labels, xai_colors = [], [], []
        for key, lbl, col in xai_keys:
            if key.startswith("1 - "):
                val = 1.0 - xai_summary.get(key[4:], 0.0)
            else:
                val = xai_summary.get(key, 0.0)
            xai_vals.append(val)
            xai_labels.append(lbl)
            xai_colors.append(col)
        xi = np.arange(len(xai_vals))
        bars = ax2.bar(xi, xai_vals, color=xai_colors, width=0.5)
        ax2.set_ylim(0, 1.15)
        ax2.set_xticks(xi)
        ax2.set_xticklabels(xai_labels, fontsize=11)
        ax2.set_ylabel("Score")
        ax2.set_title("XAI attack effectiveness\n(all bars: higher = attack more successful)")
        ax2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    fig.tight_layout()
    return fig


def _heatmap_to_rgb(heatmap: np.ndarray, cmap: str = "hot") -> np.ndarray:
    """Normalize a 2-D heatmap → [H, W, 3] float32 in [0, 1]."""
    lo, hi = heatmap.min(), heatmap.max()
    norm = (heatmap - lo) / (hi - lo + 1e-8)
    return plt.get_cmap(cmap)(norm)[:, :, :3].astype(np.float32)


def _save_sample_images(
    sample_dir: Path,
    orig_wav: torch.Tensor,
    adv_wav: torch.Tensor,
    delta_wav: torch.Tensor,
    cam_orig: np.ndarray,
    cam_adv: np.ndarray,
    sample_rate: int,
    stem: str,
    label: int,
    pred_orig: int,
    pred_adv: int,
) -> None:
    """Save spectrograms, CAM heatmaps, overlays, and a summary panel as PNGs."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    def _spec_hwc(wav: torch.Tensor, cmap: str = "inferno") -> np.ndarray:
        return _make_spectrogram_image(wav, sample_rate, max_seconds=None, cmap=cmap).permute(1, 2, 0).numpy()

    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        """[H,W,3] float RGB → [H,W,3] luminance grayscale."""
        g = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        return np.stack([g, g, g], axis=-1)

    def _cam_hwc(cam: np.ndarray, ref_shape: tuple) -> np.ndarray:
        if cam.ndim == 1:
            cam = cam[np.newaxis, :]   # [T] → [1, T]: treat as freq=1 time map
        t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=ref_shape, mode="bilinear", align_corners=False)
        return _heatmap_to_rgb(t.squeeze().numpy(), cmap="hot")

    # Original-colour spectrograms (inferno) — kept for individual files
    spec_orig  = _spec_hwc(orig_wav)
    spec_adv   = _spec_hwc(adv_wav)
    # Delta uses the same "hot" colormap as the CAM heatmaps
    spec_delta = _spec_hwc(delta_wav, cmap="hot")
    H, W = spec_orig.shape[:2]

    # Grayed spectrograms — used in the panel so CAMs stand out clearly
    spec_orig_gray = _to_gray(spec_orig)
    spec_adv_gray  = _to_gray(spec_adv)

    hmap_orig = _cam_hwc(cam_orig, (H, W))
    hmap_adv  = _cam_hwc(cam_adv,  (H, W))
    ov_orig   = np.clip(0.55 * spec_orig_gray + 0.45 * hmap_orig, 0, 1)
    ov_adv    = np.clip(0.55 * spec_adv_gray  + 0.45 * hmap_adv,  0, 1)

    # Individual files: original-colour + gray variants + hot delta
    plt.imsave(str(sample_dir / "spectrogram_original.png"),      spec_orig)
    plt.imsave(str(sample_dir / "spectrogram_adversarial.png"),   spec_adv)
    plt.imsave(str(sample_dir / "spectrogram_original_gray.png"), spec_orig_gray)
    plt.imsave(str(sample_dir / "spectrogram_adversarial_gray.png"), spec_adv_gray)
    plt.imsave(str(sample_dir / "spectrogram_delta.png"),         spec_delta)
    plt.imsave(str(sample_dir / "heatmap_original.png"),          hmap_orig)
    plt.imsave(str(sample_dir / "heatmap_adversarial.png"),       hmap_adv)
    plt.imsave(str(sample_dir / "overlay_original.png"),          ov_orig)
    plt.imsave(str(sample_dir / "overlay_adversarial.png"),       ov_adv)

    true_lbl  = "real" if label == 0 else "fake"
    porig_lbl = "real" if pred_orig == 0 else "fake"
    padv_lbl  = "real" if pred_adv  == 0 else "fake"

    # Panel: 2×3 grid
    #   col 1 — gray spectrogram   col 2 — CAM heatmap   col 3 — CAM overlaid on gray spec
    fig, axes = plt.subplots(2, 3, figsize=(18, 6))
    for ax, img, ttl in zip(axes[0],
                             [spec_orig_gray, hmap_orig, ov_orig],
                             ["Original spectrogram (gray)", "Original CAM", "Original overlay"]):
        ax.imshow(img, aspect="auto", origin="upper"); ax.set_title(ttl, fontsize=9); ax.axis("off")
    for ax, img, ttl in zip(axes[1],
                             [spec_adv_gray, hmap_adv, ov_adv],
                             ["Adversarial spectrogram (gray)", "Adversarial CAM", "Adversarial overlay"]):
        ax.imshow(img, aspect="auto", origin="upper"); ax.set_title(ttl, fontsize=9); ax.axis("off")
    fig.suptitle(
        f"{stem}  |  true={true_lbl}  "
        f"pred_orig={porig_lbl}  pred_adv={padv_lbl}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(sample_dir / "panel.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


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

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            m["pesq"] = round(compute_pesq(orig_16k, adv_16k, sr=PESQ_SR), 6)
    except Exception as e:
        m["pesq"] = None
        print(f"  [warn] PESQ [{i}]: {e}")

    try:
        m["stoi"] = round(compute_stoi(orig_16k, adv_16k, sr=PESQ_SR), 6)
    except Exception as e:
        m["stoi"] = None
        print(f"  [warn] STOI [{i}]: {e}")

    try:
        from visqol import VisqolApi
        api = VisqolApi()
        api.create(mode="speech")
        result = api.measure_from_arrays(
            orig_16k.numpy().astype(np.float64),
            adv_16k.numpy().astype(np.float64),
            sample_rate=PESQ_SR,
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
        vals = [m[key] for m in audio_metrics
                if m.get(key) is not None and not np.isnan(m[key])]
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


# ── Lightning attack module ───────────────────────────────────────────────────

class AttackModule(LightningModule):
    """LightningModule wrapper around the per-sample perceptual XAI attack.

    Used exclusively with Trainer.predict().  Each predict_step processes one
    audio file: load → attack → GradCAM → psychoacoustics → save artifacts →
    return metrics dict.  Memory is explicitly freed in the finally block before
    the next sample.
    """

    def __init__(
        self,
        args,
        infer_model:  torch.nn.Module,
        attack_model: torch.nn.Module,
        sample_rate:  int,
        cfg:          AttackConfig,
        clip_len:     int,
        hop_len:      int,
        sw_micro_bs:  int,
        log_path:     Path,
    ) -> None:
        super().__init__()
        self.args         = args
        self.infer_model  = infer_model
        self.attack_model = attack_model
        self.sample_rate  = sample_rate
        self.cfg          = cfg
        self.clip_len     = clip_len
        self.hop_len      = hop_len
        self.sw_micro_bs  = sw_micro_bs
        self.log_path     = log_path
        self.audio_dir    = log_path / "audio"

    # Lightning wraps predict_step in torch.no_grad() even when
    # inference_mode=False.  GradCAM needs real gradients, so we
    # re-enable gradient tracking for the entire step via this decorator.
    @torch.enable_grad()
    def predict_step(self, batch, batch_idx: int) -> dict:
        path_strs, labels_t = batch
        path      = Path(path_strs[0])
        label     = int(labels_t[0].item())
        index_var = batch_idx   # progress tracker: which sample we are attacking
        dev       = str(self.device)

        _gpu_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"\n{'='*60}")
        print(f"  [index_var={index_var}] {path.name}  "
              f"(label={'real' if label==0 else 'fake'})  GPU: {_gpu_used:.2f} GiB")
        print(f"{'='*60}")

        max_oom_retries = getattr(self.args, 'oom_retries', 3)
        result: dict = {"index": index_var, "stem": path.stem, "label": label, "ok": False}

        for oom_attempt in range(max_oom_retries):
            if oom_attempt > 0:
                _gpu_now = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(f"\n  [RETRY {oom_attempt}/{max_oom_retries - 1}] "
                      f"index_var={index_var}  GPU now: {_gpu_now:.2f} GiB")

            x_single = x_clip = x_adv_clip = None
            x_adv_full = delta_full = delta_clip = None
            orig_cpu = adv_cpu = orig_save = adv_save = dlt_save = None
            pred_orig = prob_orig = pred_adv = prob_adv = None
            cos_sim_t = pred_pres_t = None
            cam_orig_np: np.ndarray | None = None
            cam_adv_np:  np.ndarray | None = None
            history: list[dict] = []

            try:
                if self.args.full_audio:
                    x_single = load_full_waveform(path, self.sample_rate)
                else:
                    x_single = load_waveform(path, self.sample_rate, self.clip_len)

                x_clip = x_single[:self.clip_len].unsqueeze(0).to(dev)
                pred_orig, prob_orig = predict_batch(self.infer_model, x_clip)

                if self.args.full_audio:
                    x_adv_full, delta_full, win_result = _attack_full_audio_sample(
                        self.args.model_type, self.attack_model, x_single, self.cfg,
                        self.clip_len, self.hop_len, self.sw_micro_bs, dev,
                    )
                    x_adv_clip  = x_adv_full[:self.clip_len].unsqueeze(0).to(dev)
                    cos_sim_t   = win_result.cosine_similarity
                    pred_pres_t = win_result.prediction_preserved
                    history     = win_result.history
                    # Aggregate all N windows by concatenating along the time axis.
                    # win_result.cam_original shape: [N, T_tokens] (1-D CAM per window)
                    # or [N, F, T_tokens] (2-D feature-map CAM).
                    _c_o = win_result.cam_original.detach().cpu().numpy()
                    _c_a = win_result.cam_adv.detach().cpu().numpy()
                    if _c_o.ndim == 2:          # [N, T] → [N*T]
                        cam_orig_np = _c_o.reshape(-1)
                        cam_adv_np  = _c_a.reshape(-1)
                    else:                       # [N, F, T] → [F, N*T]
                        cam_orig_np = np.concatenate(list(_c_o), axis=-1)
                        cam_adv_np  = np.concatenate(list(_c_a), axis=-1)
                    del win_result, _c_o, _c_a
                else:
                    gradcam = _make_gradcam(self.args.model_type, self.attack_model)
                    res     = perceptual_xai_attack(self.attack_model, x_clip, self.cfg, gradcam=gradcam)
                    x_adv_clip  = res.x_adv
                    delta_clip  = res.delta.squeeze(0).detach().cpu()
                    cos_sim_t   = res.cosine_similarity
                    pred_pres_t = res.prediction_preserved
                    history     = res.history
                    cam_orig_np = res.cam_original[0].detach().cpu().numpy()
                    cam_adv_np  = res.cam_adv[0].detach().cpu().numpy()
                    del res, gradcam

                pred_adv, prob_adv = predict_batch(self.infer_model, x_adv_clip)

                orig_cpu  = x_clip.squeeze(0).detach().cpu()
                adv_cpu   = x_adv_clip.squeeze(0).detach().cpu()
                cos_mean  = float(cos_sim_t.mean().item())
                pred_pres = int(pred_pres_t.all().item())

                if self.args.full_audio:
                    delta_linf = float(delta_full.abs().max().item())
                    orig_save  = x_single.detach().cpu()
                    adv_save   = x_adv_full.detach().cpu()
                    dlt_save   = delta_full.detach().cpu()
                else:
                    delta_linf = float(delta_clip.abs().max().item())
                    orig_save  = orig_cpu
                    adv_save   = adv_cpu
                    dlt_save   = delta_clip

                if cam_orig_np is not None:
                    cam_o_t  = torch.from_numpy(cam_orig_np).unsqueeze(0)
                    cam_a_t  = torch.from_numpy(cam_adv_np).unsqueeze(0)
                    over_val = float(topk_overlap(cam_o_t, cam_a_t, k_frac=0.1).item())
                else:
                    over_val = 0.0

                print("  Computing perceptual metrics...")
                psy = _psychoacoustic_one(index_var, orig_cpu, adv_cpu, self.sample_rate)
                psy_str = "  ".join(
                    f"{k.upper()}={v:.3f}" for k, v in psy.items() if v is not None
                )
                print(f"  {psy_str}")

                sample_metrics = {
                    "index":          index_var,
                    "stem":           path.stem,
                    "label":          label,
                    "label_str":      "real" if label == 0 else "fake",
                    "pred_orig":      int(pred_orig.item()),
                    "prob_orig":      round(float(prob_orig.item()), 6),
                    "pred_adv":       int(pred_adv.item()),
                    "prob_adv":       round(float(prob_adv.item()), 6),
                    "correct_orig":   int(pred_orig.item() == label),
                    "correct_adv":    int(pred_adv.item() == label),
                    "pred_preserved": pred_pres,
                    "cos_sim":        round(cos_mean, 6),
                    "top10_overlap":  round(over_val, 6),
                    "delta_linf":     round(delta_linf, 6),
                    **{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in psy.items()},
                    "ok": True,
                }

                sample_json = self.log_path / f"sample_{index_var:04d}_{path.stem}.json"
                sample_json.write_text(json.dumps(sample_metrics, indent=2))
                print(f"  Saved metrics → {sample_json.name}")

                sample_audio_dir = self.audio_dir / path.stem
                sample_audio_dir.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(sample_audio_dir / "original.wav"),
                                orig_save.unsqueeze(0), self.sample_rate)
                torchaudio.save(str(sample_audio_dir / "adversarial.wav"),
                                adv_save.unsqueeze(0), self.sample_rate)
                torchaudio.save(str(sample_audio_dir / "delta.wav"),
                                dlt_save.unsqueeze(0), self.sample_rate)
                print(f"  Saved audio   → {sample_audio_dir}")

                tb = self.logger.experiment
                tb.add_image(f"spectrogram/{path.stem}/original",
                             _make_spectrogram_image(orig_cpu, self.sample_rate), 0)
                tb.add_image(f"spectrogram/{path.stem}/adversarial",
                             _make_spectrogram_image(adv_cpu, self.sample_rate), 0)
                tb.add_image(f"spectrogram/{path.stem}/delta",
                             _make_spectrogram_image(dlt_save[:self.clip_len], self.sample_rate), 0)
                if index_var < 10:
                    tb.add_audio(f"audio/{path.stem}/original",
                                 orig_cpu.unsqueeze(0), sample_rate=self.sample_rate)
                    tb.add_audio(f"audio/{path.stem}/adversarial",
                                 adv_cpu.unsqueeze(0), sample_rate=self.sample_rate)

                if cam_orig_np is not None:
                    hmap_dir = self.log_path / "heatmaps" / path.stem
                    hmap_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(hmap_dir / "original.npy"),    cam_orig_np)
                    np.save(str(hmap_dir / "adversarial.npy"), cam_adv_np)
                    _save_sample_images(
                        sample_dir  = self.log_path / "images" / path.stem,
                        orig_wav    = orig_save if self.args.full_audio else orig_cpu,
                        adv_wav     = adv_save  if self.args.full_audio else adv_cpu,
                        delta_wav   = dlt_save,
                        cam_orig    = cam_orig_np,
                        cam_adv     = cam_adv_np,
                        sample_rate = self.sample_rate,
                        stem        = path.stem,
                        label       = label,
                        pred_orig   = int(pred_orig.item()),
                        pred_adv    = int(pred_adv.item()),
                    )

                result = {**sample_metrics, "history": history}
                break  # ── success: exit OOM retry loop ──────────────────────

            except torch.cuda.OutOfMemoryError as oom_e:
                print(f"\n  [OOM] index_var={index_var}, "
                      f"attempt {oom_attempt + 1}/{max_oom_retries}: {oom_e}")

            except Exception as exc:
                import traceback as _tb
                print(f"  [ERROR] index_var={index_var} ({path.name}): {exc}")
                _tb.print_exc()
                break  # non-OOM errors: skip retries

            finally:
                # Runs on every attempt (success, OOM, or other error) before
                # the loop body continues or exits — ensures GPU is always freed.
                x_single    = None
                x_clip      = None
                x_adv_clip  = None
                x_adv_full  = None
                delta_full  = None
                delta_clip  = None
                orig_cpu    = None
                adv_cpu     = None
                orig_save   = None
                adv_save    = None
                dlt_save    = None
                pred_orig   = None
                prob_orig   = None
                pred_adv    = None
                prob_adv    = None
                cos_sim_t   = None
                pred_pres_t = None
                cam_orig_np = None
                cam_adv_np  = None
                self.attack_model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # ── reached only after OOM (success/non-OOM both use break) ─────
            if oom_attempt < max_oom_retries - 1:
                _wait = 2 * (oom_attempt + 1)
                print(f"  Waiting {_wait}s before retry...")
                time.sleep(_wait)
            else:
                print(f"  [OOM] All {max_oom_retries} attempts exhausted for "
                      f"index_var={index_var} — skipping sample")

        # GPU stats logged AFTER cleanup so the numbers reflect actual freed state.
        # DeviceStatsMonitor only fires on training hooks, so we log manually.
        if torch.cuda.is_available():
            tb = self.logger.experiment
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_res   = torch.cuda.memory_reserved()  / 1e9
            mem_peak  = torch.cuda.max_memory_allocated() / 1e9
            tb.add_scalar("gpu/memory_allocated_GiB", mem_alloc, index_var)
            tb.add_scalar("gpu/memory_reserved_GiB",  mem_res,   index_var)
            tb.add_scalar("gpu/memory_peak_GiB",      mem_peak,  index_var)
            torch.cuda.reset_peak_memory_stats()
            print(f"  GPU post-cleanup: allocated={mem_alloc:.2f} GiB  "
                  f"reserved={mem_res:.2f} GiB  peak={mem_peak:.2f} GiB")

        return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()

    p.add_argument("--config", default=None, metavar="YAML",
                   help="YAML config file; reads [model], [predict.split], [attack] sections")

    # Model
    p.add_argument("--model-type", required=False, choices=["sonics", "ast", "vggish"])
    p.add_argument("--model-id",   default="awsaf49/sonics-spectttra-gamma-120s",
                   help="HuggingFace repo ID (sonics only)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Lightning .ckpt file (ast / vggish)")

    # Data
    p.add_argument("--data-root",    type=Path, required=False)
    p.add_argument("--split",        default="test")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate",  type=int, default=16_000,
                   help="Overridden for ast/vggish automatically")
    p.add_argument("--seed",         type=int, default=42)

    # Attack
    p.add_argument("--n-attack-samples", type=int, default=None,
                   help="Number of samples to attack (balanced real/fake). "
                        "Required: set via --config YAML (attack.n_attack_samples) "
                        "or pass explicitly.")
    p.add_argument("--attack-micro-batch", type=int, default=None,
                   help="Number of windows processed on GPU simultaneously in "
                        "full-audio mode. Default 1 (safest). Increase if VRAM allows.")
    p.add_argument("--n-steps",          type=int, default=50,
                   help="Adam optimisation steps for the attack")
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--lambda-aud",       type=float, default=1.0)
    p.add_argument("--lambda-pred",      type=float, default=100.0)
    p.add_argument("--pred-margin",      type=float, default=1.0,
                   help="Hinge margin for prediction-preserving loss. "
                        "Lower values (e.g. 0.0) activate the penalty earlier "
                        "and leave less room for prediction flips.")
    p.add_argument("--oom-retries",      type=int, default=3,
                   help="Max GPU OOM retries per sample before skipping it. "
                        "On each retry GPU memory is fully freed and re-allocated.")

    # Sliding-window full-audio mode
    p.add_argument("--full-audio",       action="store_true", default=False,
                   help="Attack the full audio length via a sliding window instead "
                        "of a fixed clip. Each file is split into clip_seconds windows "
                        "with window-hop-seconds stride, attacked independently, and "
                        "stitched back with overlap-add.")
    p.add_argument("--window-hop-seconds", type=float, default=None,
                   help="Hop size in seconds between consecutive windows. "
                        "Defaults to --clip-seconds (no overlap). "
                        "Set smaller (e.g. half of clip_seconds) for 50%% overlap "
                        "with Hann-window blending at boundaries.")

    # Batching: split n_attack_samples into N equal chunks and process one per run
    p.add_argument("--n-batches",    type=int, default=1,
                   help="Total number of batches the sample list is divided into.")
    p.add_argument("--batch-index",  type=int, default=0,
                   help="0-based index of the batch to process in this run.")

    # Output
    p.add_argument("--run-name", default=None,
                   help="Override the auto-generated TensorBoard run name. "
                        "Set the same value for all batches to write into one directory.")
    p.add_argument("--log-dir", type=Path, default=Path("runs/attack"))
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")

    if _pre_args.config:
        from audio_xai.hparams import attack_defaults
        p.set_defaults(**attack_defaults(_pre_args.config))

    args = p.parse_args()

    if args.model_type is None:
        p.error("--model-type is required (or set model.name in your --config YAML)")
    if args.data_root is None:
        p.error("--data-root is required (or set data.root in your --config YAML)")
    if args.n_attack_samples is None:
        p.error("--n-attack-samples is required (or set attack.n_attack_samples in your --config YAML)")

    print(f"Device : {args.device}")
    print(f"Model  : {args.model_type}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    infer_model, attack_model, sample_rate = load_model(args)
    clip_len = int(args.clip_seconds * sample_rate)
    print(f"Sample rate: {sample_rate} Hz  |  clip: {args.clip_seconds}s ({clip_len} samples)")

    # ── 2. Load split ─────────────────────────────────────────────────────────
    samples = load_split(args.data_root, args.split or None, seed=args.seed)
    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    print(f"\nSplit '{args.split}': {len(samples)} files  ({n_real} real, {n_fake} fake)")
    if not samples:
        print("No files found — check --data-root and --split.")
        return

    # ── 3. Select balanced sample list (no audio loaded yet) ──────────────────
    print(f"\n── Perceptual XAI attack  "
          f"({args.n_attack_samples} samples, {args.n_steps} steps) ───────")
    atk_paths, gt_labels_list = _select_balanced_batch(
        samples, args.n_attack_samples, seed=args.seed
    )

    if args.n_batches > 1:
        per_batch = len(atk_paths) // args.n_batches
        start = args.batch_index * per_batch
        end   = start + per_batch if args.batch_index < args.n_batches - 1 else len(atk_paths)
        atk_paths      = atk_paths[start:end]
        gt_labels_list = gt_labels_list[start:end]
        print(f"Batch {args.batch_index + 1}/{args.n_batches}: "
              f"samples [{start}, {end}) of {args.n_attack_samples}")

    n_sel_real = sum(1 for l in gt_labels_list if l == 0)
    n_sel_fake = sum(1 for l in gt_labels_list if l == 1)
    print(f"Selected {len(atk_paths)} samples  ({n_sel_real} real, {n_sel_fake} fake)")

    # ── 4. Setup paths + Lightning TensorBoard logger ─────────────────────────
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else args.model_id.replace("/", "_")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = args.run_name or f"{args.model_type}_{ckpt_tag}_{args.split}_{ts}"

    tb_logger = TensorBoardLogger(
        save_dir=str(args.log_dir),
        name=run_name,
        version="",
    )
    log_path = Path(tb_logger.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # ── 4b. Resume: skip already-processed samples ────────────────────────────
    audio_out_dir = log_path / "audio"
    done_stems = {
        p.parent.name
        for p in audio_out_dir.glob("*/adversarial.wav")
    } if audio_out_dir.exists() else set()

    if done_stems:
        before = len(atk_paths)
        filtered = [(p, l) for p, l in zip(atk_paths, gt_labels_list)
                    if Path(p).stem not in done_stems]
        atk_paths, gt_labels_list = (map(list, zip(*filtered)) if filtered
                                     else ([], []))
        skipped = before - len(atk_paths)
        print(f"\nResume: {skipped}/{before} already done, {len(atk_paths)} remaining")
        for stem in sorted(done_stems):
            print(f"  [done]  {stem}")
        for p in atk_paths:
            print(f"  [todo]  {Path(p).stem}")
    else:
        print(f"\nResume: no prior output found — processing all {len(atk_paths)} samples")

    if not atk_paths:
        print("All samples already processed — nothing to do.")
        return

    # ── 5. Attack config ───────────────────────────────────────────────────────
    cfg = AttackConfig(
        n_steps=args.n_steps,
        lr=args.lr,
        lambda_audibility=args.lambda_aud,
        lambda_pred=args.lambda_pred,
        pred_margin=args.pred_margin,
        log_every=1,
        sample_rate=sample_rate,
    )
    hop_len = int((args.window_hop_seconds or args.clip_seconds) * sample_rate)
    # Full-audio mode MUST use 1 window at a time: the YAML micro_batch (e.g. 16)
    # was meant for clipped-mode sample batching, not for stacking 16 windows
    # simultaneously through a second-order GradCAM backward — that fills 39 GiB.
    sw_micro_bs = args.attack_micro_batch or 1

    if args.full_audio:
        overlap_pct = (1.0 - hop_len / clip_len) * 100.0
        print(f"  Full-audio mode: window={args.clip_seconds}s  "
              f"hop={hop_len/sample_rate:.2f}s  overlap={overlap_pct:.0f}%  "
              f"micro_bs={sw_micro_bs}")

    # ── 6. Dataset + DataLoader ────────────────────────────────────────────────
    dataset    = AudioPathDataset(atk_paths, gt_labels_list)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # ── 7. Attack module ───────────────────────────────────────────────────────
    module = AttackModule(
        args         = args,
        infer_model  = infer_model,
        attack_model = attack_model,
        sample_rate  = sample_rate,
        cfg          = cfg,
        clip_len     = clip_len,
        hop_len      = hop_len,
        sw_micro_bs  = sw_micro_bs,
        log_path     = log_path,
    )

    # ── 8. Trainer ─────────────────────────────────────────────────────────────
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(SLURMEnvironment(auto_requeue=True))

    trainer = Trainer(
        accelerator          = "gpu" if "cuda" in args.device else "cpu",
        devices              = 1,
        logger               = tb_logger,
        callbacks            = [DeviceStatsMonitor()],
        enable_checkpointing = False,
        inference_mode       = False,   # GradCAM needs gradients during predict
        plugins              = plugins or None,
    )

    # ── 9. Run predict (one sample at a time via DataLoader batch_size=1) ──────
    t_total     = time.time()
    raw_results = trainer.predict(module, dataloaders=dataloader)
    elapsed     = time.time() - t_total

    # ── 10. Aggregate results ──────────────────────────────────────────────────
    results = [r for r in (raw_results or []) if r.get("ok")]
    n_done  = len(results)
    print(f"\nAll {n_done}/{len(atk_paths)} samples done in {elapsed:.1f}s  "
          f"({elapsed / max(n_done, 1):.1f}s / sample)")

    if not results:
        print("No samples processed successfully.")
        return

    all_labels        = [r["label"]         for r in results]
    all_preds_orig    = [r["pred_orig"]      for r in results]
    all_probs_orig    = [r["prob_orig"]      for r in results]
    all_preds_adv     = [r["pred_adv"]       for r in results]
    all_probs_adv     = [r["prob_adv"]       for r in results]
    all_cos_sims      = [r["cos_sim"]        for r in results]
    all_top10_overlap = [r["top10_overlap"]  for r in results]
    all_pred_pres     = [r["pred_preserved"] for r in results]
    all_delta_linf    = [r["delta_linf"]     for r in results]
    all_audio_quality = [
        {k: r.get(k) for k in ("pesq", "stoi", "visqol", "peaq", "zimtohrli")}
        for r in results
    ]
    all_histories = [r.get("history", []) for r in results]

    # ── Aggregate classification metrics ───────────────────────────────────────
    labels_np     = np.array(all_labels)
    preds_orig_np = np.array(all_preds_orig)
    probs_orig_np = np.array(all_probs_orig)
    preds_adv_np  = np.array(all_preds_adv)
    probs_adv_np  = np.array(all_probs_adv)

    orig_atk_metrics = compute_metrics(labels_np, preds_orig_np, probs_orig_np)
    adv_metrics      = compute_metrics(labels_np, preds_adv_np,  probs_adv_np)
    _print_metrics("Original (attack batch)", orig_atk_metrics)
    _print_metrics("Adversarial predictions", adv_metrics)

    mean_cos       = float(np.mean(all_cos_sims))      if all_cos_sims      else 0.0
    mean_over      = float(np.mean(all_top10_overlap)) if all_top10_overlap else 0.0
    mean_linf      = float(np.mean(all_delta_linf))    if all_delta_linf    else 0.0
    pred_pres_frac = float(np.mean(all_pred_pres))     if all_pred_pres     else 0.0

    print(f"\n  CAM cosine sim  : mean={mean_cos:.4f}  (lower = more explanation change)")
    print(f"  Top-10% overlap : mean={mean_over:.4f}  (lower = more disagreement)")
    print(f"  δ L∞ mean       : {mean_linf:.5f}")
    print(f"  Pred preserved  : {sum(all_pred_pres)}/{n_done}")

    print(f"\n── Δ (adversarial − original) ──────────────────────────────────────")
    for k in ("accuracy", "f1", "auc"):
        ov, av = orig_atk_metrics[k], adv_metrics[k]
        print(f"  Δ{k:10s}: {av - ov:+.4f}  ({ov:.4f} → {av:.4f})")

    # ── Perceptual quality summary ─────────────────────────────────────────────
    print("\n── Perceptual audio quality metrics (orig → adv) ───────────────────")
    perceptual_means: dict[str, float] = {}
    for key in ("pesq", "stoi", "visqol", "peaq", "zimtohrli"):
        vals = [m[key] for m in all_audio_quality
                if m.get(key) is not None
                and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if vals:
            print(f"  {key.upper():10s}: mean={np.mean(vals):.4f}  "
                  f"[{min(vals):.3f} … {max(vals):.3f}]  n={len(vals)}")
            perceptual_means[f"mean_{key}"] = float(np.mean(vals))

    atk_summary = {
        "mean_cos_sim":        mean_cos,
        "top10_overlap":       mean_over,
        "delta_linf":          mean_linf,
        "pred_preserved_frac": pred_pres_frac,
    }

    # ── Summary JSON ───────────────────────────────────────────────────────────
    summary: dict = {
        "run": {
            "model_type":         args.model_type,
            "model_id":           args.model_id if args.model_type == "sonics" else None,
            "checkpoint":         str(args.checkpoint) if args.checkpoint else None,
            "split":              args.split,
            "clip_seconds":       args.clip_seconds,
            "sample_rate":        sample_rate,
            "n_attack_samples":   n_done,
            "n_steps":            args.n_steps,
            "lambda_aud":         args.lambda_aud,
            "lambda_pred":        args.lambda_pred,
            "pred_margin":        args.pred_margin,
            "full_audio":         args.full_audio,
            "window_hop_seconds": args.window_hop_seconds,
        },
        "original_metrics_atk_batch": orig_atk_metrics,
        "adversarial_metrics":        adv_metrics,
        "attack_summary":             atk_summary,
        "perceptual_means":           perceptual_means,
    }
    out_json = log_path / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nResults → {out_json}")

    # ── TensorBoard aggregate writes ───────────────────────────────────────────
    # trainer.predict() closes tb_logger internally, so open a fresh writer that
    # appends events to the same log directory.
    writer = SummaryWriter(log_dir=str(log_path))

    def _metrics_md_table(metrics: dict, title: str) -> str:
        rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items()
                         if isinstance(v, float))
        return f"**{title}**\n\n| Metric | Value |\n|--------|-------|\n{rows}"

    fig = _metrics_bar_figure(
        orig_atk_metrics, adv_metrics,
        title=f"{args.model_type} — Classification metrics ({n_done} samples)",
        xai_summary=atk_summary,
    )
    writer.add_figure("attack/metrics_comparison", fig, 0)
    plt.close(fig)

    delta_metrics = {k: adv_metrics[k] - orig_atk_metrics[k]
                     for k in ("accuracy", "precision", "recall", "f1", "auc")
                     if k in adv_metrics and k in orig_atk_metrics}
    writer.add_text("metrics/original_atk_batch",
                    _metrics_md_table(orig_atk_metrics, "Original (attack batch)"), 0)
    writer.add_text("metrics/adversarial",
                    _metrics_md_table(adv_metrics, "Adversarial"), 0)
    writer.add_text("metrics/delta",
                    _metrics_md_table(delta_metrics, "Δ (adversarial − original)"), 0)
    writer.add_text("metrics/attack_summary",
                    _metrics_md_table(atk_summary, "Attack summary"), 0)
    if perceptual_means:
        writer.add_text("metrics/perceptual",
                        _metrics_md_table(perceptual_means, "Perceptual quality (orig → adv)"), 0)

    if all_histories:
        n_steps_hist = max(len(h) for h in all_histories)
        merged_history: list[dict] = []
        for s in range(n_steps_hist):
            entries = [h[s] for h in all_histories if s < len(h)]
            merged_history.append({
                k: sum(e[k] for e in entries) / len(entries) for k in entries[0]
            })
        for entry in merged_history:
            s = entry["step"]
            writer.add_scalar("loss/total",      entry["loss"],            s)
            writer.add_scalar("loss/explain",    entry["loss_explain"],    s)
            writer.add_scalar("loss/audibility", entry["loss_audibility"], s)
            writer.add_scalar("loss/pred",       entry["loss_pred"],       s)
            writer.add_scalar("loss/cos_sim",    entry["cos_sim"],         s)
        (log_path / "loss_history.json").write_text(json.dumps(merged_history, indent=2))

    if all_audio_quality:
        fig = _psychoacoustic_bar_figure(
            all_audio_quality,
            title=f"{args.model_type} — Perceptual Quality ({n_done} samples)",
        )
        writer.add_figure("attack/metrics_comparison_psychoacoustic", fig, 0)
        plt.close(fig)

    writer.close()
    print(f"\nTensorBoard → {log_path}")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()