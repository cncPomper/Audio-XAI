#!/usr/bin/env python3
"""
XAI Fragility Experiment — Speech Commands Dataset
====================================================
Runs the perceptual XAI attack on AST, VGGish, or SpecTra models using
Google Speech Commands audio clips. Measures:
  - Explanation fragility (GradCAM): cosine sim, top-k Jaccard, heatmap SSIM
  - Explanation fragility (LRP):     cosine sim, top-k Jaccard, heatmap SSIM
  - Perturbation budget:   L∞, RMS, SNR
  - Audio quality BEFORE attack: RMS level, peak dBFS of original waveform
  - Audio quality AFTER attack:  PESQ, STOI, SNR, LSD (original vs adversarial)

Usage (one model at a time):
  python scripts/run_speech_experiment.py \\
      --speech-dir speech_commands_samples \\
      --model ast \\
      --out-dir reports/speech_ast

  python scripts/run_speech_experiment.py \\
      --speech-dir speech_commands_samples \\
      --model vggish \\
      --vggish-ckpt /path/to/vggish_model.ckpt \\
      --out-dir reports/speech_vggish

  python scripts/run_speech_experiment.py \\
      --speech-dir speech_commands_samples \\
      --model spectra \\
      --out-dir reports/speech_spectra

SLURM example:
  sbatch --gres=gpu:1 --mem=32G --time=4:00:00 \\
      --wrap="python scripts/run_speech_experiment.py \\
          --speech-dir /path/to/speech_commands_samples \\
          --model ast --n-samples 200 --out-dir reports/speech_ast"
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_xai.attacks.perceptual_xai_attack import (
    AttackConfig,
    heatmap_ssim,
    perceptual_xai_attack,
    topk_overlap,
)
from audio_xai.data.speech_commands_dataset import SpeechCommandsConfig, SpeechCommandsDataset
from audio_xai.models.ast_binary import ASTBinary
from audio_xai.models.vggish_binary import VGGishBinary
from audio_xai.models.wav2vec2_binary import Wav2Vec2Binary
from audio_xai.xai.lrp import make_lrp

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(name: str, checkpoint: Path | None, vggish_ckpt: str | None):
    if name == "ast":
        model = ASTBinary(pretrained=True)
    elif name == "vggish":
        model = VGGishBinary(vggish_ckpt=vggish_ckpt)
    elif name == "spectra":
        model = Wav2Vec2Binary(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose ast | vggish | spectra.")

    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        from audio_xai.models.lit_module import RealFakeLitModule
        lit = RealFakeLitModule.load_from_checkpoint(
            str(checkpoint), model=model, strict=False, map_location="cpu"
        )
        model = lit.model
        print(f"[checkpoint] loaded from {checkpoint}")
    else:
        print(f"[model] {name} — pretrained weights")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Audio quality helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rms_dbfs(wav: torch.Tensor) -> float:
    """RMS amplitude in dBFS. -inf when signal is silent."""
    rms = wav.pow(2).mean().sqrt().item()
    if rms < 1e-10:
        return float("-inf")
    return 20.0 * math.log10(rms)


def _peak_dbfs(wav: torch.Tensor) -> float:
    peak = wav.abs().max().item()
    if peak < 1e-10:
        return float("-inf")
    return 20.0 * math.log10(peak)


def _snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    sig_pow = signal.pow(2).mean().item()
    noi_pow = noise.pow(2).mean().item()
    if noi_pow < 1e-14:
        return float("inf")
    return 10.0 * math.log10(sig_pow / (noi_pow + 1e-14))


def _pesq_score(orig: torch.Tensor, adv: torch.Tensor, sr: int = 16_000) -> float | None:
    """PESQ wideband score: original (reference) vs adversarial (degraded).

    Returns None if the clip is too short for PESQ (< 0.5 s).
    """
    try:
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
        if orig.shape[-1] < sr // 2:
            return None
        pesq_fn = PerceptualEvaluationSpeechQuality(sr, "wb")
        a = orig.squeeze().cpu().float()
        b = adv.squeeze().cpu().float()
        min_len = min(a.shape[-1], b.shape[-1])
        return float(pesq_fn(a[:min_len], b[:min_len]).item())
    except Exception:
        return None


def _stoi_score(orig: torch.Tensor, adv: torch.Tensor, sr: int = 16_000) -> float | None:
    """STOI intelligibility score: original vs adversarial."""
    try:
        from pystoi import stoi
        a = orig.squeeze().cpu().numpy().astype(float)
        b = adv.squeeze().cpu().numpy().astype(float)
        min_len = min(len(a), len(b))
        return float(stoi(a[:min_len], b[:min_len], sr, extended=False))
    except Exception:
        return None


def _lsd_score(orig: torch.Tensor, adv: torch.Tensor) -> float:
    """Log-spectral distance between original and adversarial waveforms."""
    import librosa
    eps = 1e-8
    a = orig.squeeze().cpu().numpy().astype(float)
    b = adv.squeeze().cpu().numpy().astype(float)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    Sa = np.abs(librosa.stft(a))
    Sb = np.abs(librosa.stft(b))
    min_frames = min(Sa.shape[1], Sb.shape[1])
    log_Sa = np.log10(Sa[:, :min_frames] ** 2 + eps)
    log_Sb = np.log10(Sb[:, :min_frames] ** 2 + eps)
    return float(np.mean(np.sqrt(np.mean((log_Sa - log_Sb) ** 2, axis=0))))


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_convergence_plot(histories, model_name, n, out_dir):
    import matplotlib.pyplot as plt

    keys = ["loss_explain", "loss_audibility", "loss_pred", "cos_sim"]
    titles = ["Explanation loss (↓)", "Audibility loss (↓)", "Prediction loss (↓)", "Cosine similarity (↓)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, title in zip(axes.flat, keys, titles):
        curves = [[h[key] for h in hist] for hist in histories if hist]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        steps = [h["step"] for h in histories[0][:min_len]]
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(steps, mean, lw=2, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_title(title)
        ax.set_xlabel("Attack step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Attack convergence — {model_name}  (n={n})", fontsize=13)
    fig.tight_layout()
    path = out_dir / "convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot → {path}")


def _save_quality_plot(df: pd.DataFrame, model_name: str, n: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    quality_cols = [c for c in ["pesq", "stoi", "lsd", "snr_db"] if c in df.columns]
    if not quality_cols:
        return

    fig, axes = plt.subplots(1, len(quality_cols), figsize=(4 * len(quality_cols), 4))
    if len(quality_cols) == 1:
        axes = [axes]

    labels = {"pesq": "PESQ (↑ better)", "stoi": "STOI (↑ better)",
              "lsd": "LSD (↓ better)", "snr_db": "SNR dB (↑ better)"}
    colors = {"pesq": "steelblue", "stoi": "darkorange", "lsd": "firebrick", "snr_db": "seagreen"}

    for ax, col in zip(axes, quality_cols):
        vals = df[col].dropna()
        vals.hist(bins=20, ax=ax, edgecolor="black", color=colors.get(col, "grey"))
        ax.set_title(f"{labels.get(col, col)}\noriginal vs adversarial")
        ax.set_xlabel(col)

    fig.suptitle(f"Audio quality after attack — {model_name}  (n={n})", fontsize=12)
    fig.tight_layout()
    path = out_dir / "quality_after_attack.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Quality plot → {path}")


def _save_distribution_plot(df: pd.DataFrame, model_name: str, n: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    df["cosine_sim"].hist(bins=20, ax=axes[0], edgecolor="black", color="steelblue")
    axes[0].set_title("Cosine similarity (orig vs adv CAM)\nlower = more manipulated")
    df["topk_overlap_10pct"].hist(bins=20, ax=axes[1], edgecolor="black", color="darkorange")
    axes[1].set_title("Top-10% Jaccard overlap\nlower = more manipulated")
    df["heatmap_ssim"].hist(bins=20, ax=axes[2], edgecolor="black", color="seagreen")
    axes[2].set_title("Heatmap SSIM\nlower = more manipulated")
    fig.suptitle(f"Explanation-similarity distributions — {model_name}  (n={n})", fontsize=12)
    fig.tight_layout()
    path = out_dir / "distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Distributions plot → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="XAI Fragility Experiment — Speech Commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--speech-dir", type=Path, required=True,
                   help="Directory with pre-fetched speech_commands WAV files "
                        "(output of fetch_speech.py)")
    p.add_argument("--model", choices=["ast", "vggish", "spectra"], default="ast",
                   help="Model to attack")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional Lightning .ckpt to override pretrained weights")
    p.add_argument("--clip-seconds", type=float, default=1.0,
                   help="Clip duration in seconds (speech_commands = 1 s)")
    p.add_argument("--label-filter", type=str, default=None,
                   help="Restrict to one spoken word, e.g. 'yes'")
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--attack-steps", type=int, default=200)
    p.add_argument("--attack-lr", type=float, default=1e-3)
    p.add_argument("--lambda-aud", type=float, default=1.0)
    p.add_argument("--lambda-pred", type=float, default=100.0)
    p.add_argument("--linf-bound", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=Path("reports/speech_experiment"))
    p.add_argument("--vggish-ckpt", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = args.out_dir / "runs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = SpeechCommandsConfig(
        root=args.speech_dir,
        clip_seconds=args.clip_seconds,
        label_filter=args.label_filter,
    )
    dataset = SpeechCommandsDataset(ds_cfg)
    n = min(args.n_samples, len(dataset))
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=rng)[:n].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=args.device.startswith("cuda"))
    print(f"Speech Commands: {len(dataset)} files → attacking {n} samples")
    print(f"Label map: {dataset.label2idx}")

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _build_model(args.model, args.checkpoint, args.vggish_ckpt)
    model = model.to(args.device)
    model.eval()

    if hasattr(model, "backbone") and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # ── LRP explainer (computed independently of the attack) ──────────────────
    lrp = make_lrp(model)

    # ── Attack loop ───────────────────────────────────────────────────────────
    atk_cfg = AttackConfig(
        n_steps=args.attack_steps,
        lr=args.attack_lr,
        lambda_audibility=args.lambda_aud,
        lambda_pred=args.lambda_pred,
        linf_bound=args.linf_bound,
        log_every=20,
    )

    records: list[dict] = []
    histories: list[list[dict]] = []

    for i, (wav, label) in enumerate(tqdm(loader, desc=f"Attacking [{args.model}]")):
        wav = wav.to(args.device)

        # Resolve source filename from the original dataset index.
        src_filename = dataset.filename_at(indices[i])
        stem = Path(src_filename).stem   # e.g. "yes_e9c9ef6a_nohohit_0"

        # ── Quality BEFORE attack ─────────────────────────────────────────
        wav_cpu = wav.cpu()
        rms_before = _rms_dbfs(wav_cpu)
        peak_before = _peak_dbfs(wav_cpu)

        # ── LRP BEFORE attack ─────────────────────────────────────────────
        with torch.no_grad():
            lrp_orig = lrp(wav)           # [B, H, W]

        result = perceptual_xai_attack(model, wav, cfg=atk_cfg)

        # ── LRP AFTER attack ──────────────────────────────────────────────
        with torch.no_grad():
            lrp_adv = lrp(result.x_adv.to(args.device))

        lrp_cos   = topk_overlap(lrp_orig, lrp_adv, k_frac=1.0)  # cosine proxy via overlap
        # Proper cosine similarity for LRP maps
        lo_flat = lrp_orig.reshape(lrp_orig.shape[0], -1).float()
        la_flat = lrp_adv.reshape(lrp_adv.shape[0], -1).float()
        lrp_cosine = (
            (lo_flat * la_flat).sum(-1)
            / (lo_flat.norm(dim=-1) * la_flat.norm(dim=-1) + 1e-8)
        ).squeeze().item()
        lrp_overlap  = topk_overlap(lrp_orig, lrp_adv, k_frac=0.1).squeeze().item()
        lrp_ssim_val = heatmap_ssim(lrp_orig, lrp_adv).squeeze().item()

        # ── Quality AFTER attack (original vs adversarial) ────────────────
        adv_cpu = result.x_adv.cpu()
        pesq = _pesq_score(wav_cpu, adv_cpu)
        stoi = _stoi_score(wav_cpu, adv_cpu)
        lsd = _lsd_score(wav_cpu, adv_cpu)
        snr = _snr_db(wav_cpu, result.delta.cpu())

        # ── GradCAM metrics (from attack result) ──────────────────────────
        cos_sim  = result.cosine_similarity.item()
        overlap  = topk_overlap(result.cam_original, result.cam_adv, k_frac=0.1).squeeze().item()
        ssim_val = heatmap_ssim(result.cam_original, result.cam_adv).squeeze().item()
        d_linf   = result.delta.abs().max().item()
        d_rms    = result.delta.pow(2).mean().sqrt().item()

        with torch.no_grad():
            pred_orig = model(wav).argmax(-1).item()
            pred_adv  = model(result.x_adv.to(args.device)).argmax(-1).item()

        # ── Save audio with original filename ─────────────────────────────
        sr = ds_cfg.sample_rate
        torchaudio.save(str(audio_dir / f"orig_{stem}.wav"),
                        wav_cpu.unsqueeze(0), sr)
        torchaudio.save(str(audio_dir / f"adv_{stem}.wav"),
                        result.x_adv.cpu().unsqueeze(0), sr)

        records.append({
            "sample_id":   i,
            "source_file": src_filename,
            "true_label":  label.item(),
            "pred_orig":   pred_orig,
            "pred_adv":    pred_adv,
            "prediction_preserved": bool(result.prediction_preserved.item()),
            # GradCAM explanation fragility (attack targets GradCAM)
            "gradcam_cosine_sim":   round(cos_sim, 6),
            "gradcam_topk_overlap": round(overlap, 6),
            "gradcam_heatmap_ssim": round(ssim_val, 6),
            # LRP explanation fragility (collateral effect on LRP)
            "lrp_cosine_sim":       round(lrp_cosine, 6),
            "lrp_topk_overlap":     round(lrp_overlap, 6),
            "lrp_heatmap_ssim":     round(lrp_ssim_val, 6),
            # Perturbation budget
            "delta_linf": round(d_linf, 8),
            "delta_rms":  round(d_rms, 8),
            # Quality BEFORE attack (absolute, on original)
            "rms_before_dbfs":  round(rms_before, 3)  if math.isfinite(rms_before)  else None,
            "peak_before_dbfs": round(peak_before, 3) if math.isfinite(peak_before) else None,
            # Quality AFTER attack (original vs adversarial)
            "pesq":   round(pesq, 4) if pesq is not None else None,
            "stoi":   round(stoi, 4) if stoi is not None else None,
            "lsd":    round(lsd, 4),
            "snr_db": round(snr, 2)  if math.isfinite(snr)  else None,
            # Saved audio paths (relative to out_dir)
            "audio_orig": f"audio/orig_{stem}.wav",
            "audio_adv":  f"audio/adv_{stem}.wav",
        })
        # keep old aliases for backward-compat with plot helpers
        records[-1]["cosine_sim"]         = records[-1]["gradcam_cosine_sim"]
        records[-1]["topk_overlap_10pct"] = records[-1]["gradcam_topk_overlap"]
        records[-1]["heatmap_ssim"]       = records[-1]["gradcam_heatmap_ssim"]

        writer.add_text("Sample/source_file", src_filename, i)
        histories.append(result.history)

        writer.add_scalar("GradCAM/cosine_sim",       cos_sim,      i)
        writer.add_scalar("GradCAM/topk_overlap",      overlap,      i)
        writer.add_scalar("GradCAM/heatmap_ssim",      ssim_val,     i)
        writer.add_scalar("LRP/cosine_sim",            lrp_cosine,   i)
        writer.add_scalar("LRP/topk_overlap",          lrp_overlap,  i)
        writer.add_scalar("LRP/heatmap_ssim",          lrp_ssim_val, i)
        writer.add_scalar("Quality/pesq",              pesq or 0.0,  i)
        writer.add_scalar("Quality/stoi",              stoi or 0.0,  i)
        writer.add_scalar("Quality/lsd",               lsd,          i)
        writer.add_scalar("Perturbation/snr_db",       snr if math.isfinite(snr) else 0.0, i)
        writer.add_scalar("Perturbation/delta_linf",   d_linf,       i)

        del result, lrp_orig, lrp_adv

        del result, wav
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── Outputs ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = args.out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Per-sample results → {csv_path}")

    def _stats(col: str) -> dict | None:
        if col not in df.columns:
            return None
        vals = df[col].dropna()
        if vals.empty:
            return None
        return {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        }

    pres_rate = float(df["prediction_preserved"].mean())
    success_rate = float((df["prediction_preserved"] & (df["cosine_sim"] < 0.5)).mean())

    summary = {
        "model": args.model,
        "dataset": "google/speech_commands",
        "n_samples": n,
        "seed": args.seed,
        "attack_config": {
            "n_steps": atk_cfg.n_steps,
            "lr": atk_cfg.lr,
            "lambda_audibility": atk_cfg.lambda_audibility,
            "lambda_pred": atk_cfg.lambda_pred,
            "linf_bound": atk_cfg.linf_bound,
        },
        "results": {
            "prediction_preservation_rate": round(pres_rate, 4),
            "attack_success_rate": round(success_rate, 4),
            "gradcam": {
                "cosine_sim":    _stats("gradcam_cosine_sim"),
                "topk_overlap":  _stats("gradcam_topk_overlap"),
                "heatmap_ssim":  _stats("gradcam_heatmap_ssim"),
            },
            "lrp": {
                "cosine_sim":    _stats("lrp_cosine_sim"),
                "topk_overlap":  _stats("lrp_topk_overlap"),
                "heatmap_ssim":  _stats("lrp_heatmap_ssim"),
            },
            "delta_linf_mean": round(float(df["delta_linf"].mean()), 6),
            "snr_db_mean": _stats("snr_db"),
        },
        "audio_quality": {
            "before_attack": {
                "rms_dbfs": _stats("rms_before_dbfs"),
                "peak_dbfs": _stats("peak_before_dbfs"),
            },
            "after_attack": {
                "pesq": _stats("pesq"),
                "stoi": _stats("stoi"),
                "lsd": _stats("lsd"),
                "snr_db": _stats("snr_db"),
            },
        },
    }

    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")
    writer.close()

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        _save_convergence_plot(histories, args.model, n, args.out_dir)
    except Exception as exc:
        print(f"[warn] convergence plot: {exc}")
    try:
        _save_distribution_plot(df, args.model, n, args.out_dir)
    except Exception as exc:
        print(f"[warn] distributions plot: {exc}")
    try:
        _save_quality_plot(df, args.model, n, args.out_dir)
    except Exception as exc:
        print(f"[warn] quality plot: {exc}")

    # ── Console summary ───────────────────────────────────────────────────────
    r = summary["results"]
    q = summary["audio_quality"]
    sep = "═" * 66
    print(f"\n{sep}")
    print(f"  XAI Fragility — {args.model.upper()}  |  Speech Commands  (n={n})")
    print(sep)
    print(f"  Prediction preservation   {pres_rate:.1%}")
    print(f"  Attack success rate       {success_rate:.1%}  (preserved + GradCAM cos_sim < 0.5)")
    print()
    print("  GradCAM (attack target):")
    gc = r["gradcam"]
    if gc["cosine_sim"]:
        print(f"    Cosine sim   (↓)  {gc['cosine_sim']['mean']:.3f} ± {gc['cosine_sim']['std']:.3f}")
    if gc["topk_overlap"]:
        print(f"    Top-10% Jacc (↓)  {gc['topk_overlap']['mean']:.3f} ± {gc['topk_overlap']['std']:.3f}")
    if gc["heatmap_ssim"]:
        print(f"    SSIM         (↓)  {gc['heatmap_ssim']['mean']:.3f} ± {gc['heatmap_ssim']['std']:.3f}")
    print()
    print("  LRP (collateral effect of the GradCAM attack):")
    lr = r["lrp"]
    if lr["cosine_sim"]:
        print(f"    Cosine sim   (↓)  {lr['cosine_sim']['mean']:.3f} ± {lr['cosine_sim']['std']:.3f}")
    if lr["topk_overlap"]:
        print(f"    Top-10% Jacc (↓)  {lr['topk_overlap']['mean']:.3f} ± {lr['topk_overlap']['std']:.3f}")
    if lr["heatmap_ssim"]:
        print(f"    SSIM         (↓)  {lr['heatmap_ssim']['mean']:.3f} ± {lr['heatmap_ssim']['std']:.3f}")
    print(sep)
    print("  Audio quality BEFORE attack (original signal)")
    if q["before_attack"]["rms_dbfs"]:
        print(f"    RMS          {q['before_attack']['rms_dbfs']['mean']:.2f} dBFS")
    if q["before_attack"]["peak_dbfs"]:
        print(f"    Peak         {q['before_attack']['peak_dbfs']['mean']:.2f} dBFS")
    print("  Audio quality AFTER attack (original vs adversarial)")
    if q["after_attack"]["pesq"]:
        print(f"    PESQ WB      {q['after_attack']['pesq']['mean']:.3f}  (max 4.5, higher = better)")
    if q["after_attack"]["stoi"]:
        print(f"    STOI         {q['after_attack']['stoi']['mean']:.3f}  (max 1.0, higher = better)")
    if q["after_attack"]["lsd"]:
        print(f"    LSD          {q['after_attack']['lsd']['mean']:.3f}  (lower = less spectral distortion)")
    if q["after_attack"]["snr_db"]:
        print(f"    SNR          {q['after_attack']['snr_db']['mean']:.1f} dB")
    print(sep)
    print(f"  All outputs → {args.out_dir}/")


if __name__ == "__main__":
    main()
