"""Perceptual XAI attack smoke test against a pretrained sonics model.

Loads a sonics HFAudioClassifier, wraps it to satisfy the project's
AudioClassifier interface, and runs the perceptual attack on 10 samples
to confirm the pipeline is wired up correctly.

TensorBoard logs are written to --log-dir (default: runs/sonics_attack).
Launch with:  tensorboard --logdir runs/sonics_attack

Usage (athena conda env):
    /path/to/athena/bin/python scripts/sonics_attack_test.py \\
        --model-id awsaf49/sonics-SpecTTTra-Sub-2sec \\
        [--audio-dir audio_xai/data/external] \\
        [--n-steps 50] \\
        [--log-dir runs/sonics_attack] \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from audio_xai.fetching_and_metrics.peaq_implementation import peaq_like
from audio_xai.fetching_and_metrics.preprocessing_metrics import (
    CDPAM_SR,
    PESQ_SR,
    _init_cdpam,
    compute_cdpam,
    compute_pesq,
    compute_stoi,
)
from audio_xai.metrics.visqol import ViSQOL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonics import HFAudioClassifier

from audio_xai.attacks.perceptual_xai_attack import (
    AttackConfig,
    perceptual_xai_attack,
    topk_overlap,
)
from audio_xai.models.base import AudioClassifier
from audio_xai.xai.gradcam import GradCAMBase


# ── Wrapper ───────────────────────────────────────────────────────────────────

class SonicsWrapper(AudioClassifier):
    """Adapts sonics HFAudioClassifier to the project's AudioClassifier interface.

    Splits the sonics forward pass into waveform_to_features / features_to_logits
    so that Grad-CAM hooks can intercept intermediate transformer activations.
    """

    def __init__(self, sonics_model: HFAudioClassifier) -> None:
        nn.Module.__init__(self)   # bypass AudioClassifier.__init__ (abstract)
        self._m = sonics_model

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        # [B, T] → mel spectrogram → [B, 1, H, W] (resized to model's input_shape)
        spec = self._m.ft_extractor(waveform)        # [B, n_mels, n_frames]
        spec = spec.unsqueeze(1)                     # [B, 1, n_mels, n_frames]
        spec = F.interpolate(
            spec, size=tuple(self._m.input_shape), mode="bilinear", align_corners=False
        )
        return spec                                  # [B, 1, H, W]

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        # [B, 1, H, W] → transformer tokens → [B, n_classes]
        tokens = self._m.encoder(features)           # [B, N_tokens, D]
        embeds = tokens.mean(dim=1)                  # global-average-pool over tokens
        return self._m.classifier(embeds)            # [B, n_classes]

    @property
    def target_layer(self) -> nn.Module:
        # Hook the last transformer block — captures richest semantic activations.
        return self._m.encoder.transformer.blocks[-1]


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class SpecTTTraGradCAM(GradCAMBase):
    """Token-level Grad-CAM for SpecTTTra.

    SpecTTTra produces [B, N_tokens, D] with no special CLS tokens
    (N_tokens = temporal_tokens + spectral_tokens). We pool gradients across
    the embedding dimension to produce a per-token importance map [B, N_tokens].
    """

    def _build_heatmap(
        self, activations: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        # activations / gradients: [B, N_tokens, D]
        weights = gradients.mean(dim=2, keepdim=True)     # [B, N_tokens, 1]
        cam = (weights * activations).sum(dim=2)          # [B, N_tokens]
        return F.relu(cam)                                # [B, N_tokens]


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_waveforms(
    audio_dir: Path | None,
    n_samples: int,
    sample_rate: int,
    clip_seconds: float,
    device: str,
) -> torch.Tensor:
    """Return [n_samples, T] float32 waveforms on device.

    Slices non-overlapping clips from wav/flac files under audio_dir.
    Falls back to Gaussian noise when audio_dir is None or yields too few clips.
    """
    clip_len = int(clip_seconds * sample_rate)

    if audio_dir is not None:
        try:
            import torchaudio

            audio_files = (
                sorted(audio_dir.glob("**/*.wav"))
                + sorted(audio_dir.glob("**/*.mp3"))
                + sorted(audio_dir.glob("**/*.flac"))
            )
            clips: list[torch.Tensor] = []
            for path in audio_files:
                wav, sr = torchaudio.load(str(path))
                if sr != sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, sample_rate)
                wav = wav.mean(0)  # to mono
                for start in range(0, wav.shape[0] - clip_len + 1, clip_len):
                    clips.append(wav[start : start + clip_len])
                    if len(clips) >= n_samples:
                        break
                if len(clips) >= n_samples:
                    break

            if len(clips) >= n_samples:
                return torch.stack(clips[:n_samples]).to(device)
            print(f"[warn] Only {len(clips)}/{n_samples} clips from {audio_dir}; padding with noise.")
            noise = 0.01 * torch.randn(n_samples - len(clips), clip_len)
            parts = [torch.stack(clips), noise] if clips else [noise]
            return torch.cat(parts, dim=0).to(device)

        except Exception as exc:
            print(f"[warn] Audio loading failed ({exc}); using noise.")

    print(
        f"Using synthetic Gaussian noise "
        f"({n_samples} clips × {clip_len} samples @ {sample_rate} Hz)"
    )
    return 0.01 * torch.randn(n_samples, clip_len, device=device)


# ── Audio quality metrics ─────────────────────────────────────────────────────

def compute_audio_metrics(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    sample_rate: int,
    cdpam_model=None,
) -> list[dict]:
    """Compute PESQ, STOI, ViSQOL, CDPAM, PEAQ for each original/adversarial pair.

    x_orig, x_adv: [B, T] float32 at sample_rate Hz.
    Returns a list of per-sample dicts; None values indicate a metric failure.
    """
    import torchaudio

    visqol = ViSQOL(sr=PESQ_SR)
    n = x_orig.shape[0]
    all_metrics: list[dict] = []

    for i in range(n):
        orig = x_orig[i].cpu()
        adv = x_adv[i].cpu()
        m: dict = {}

        # ── resample to 16 kHz for PESQ / STOI / ViSQOL ─────────────────────
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
            m["visqol"] = round(float(visqol.evaluate(orig_np, adv_np)), 6)
        except Exception as e:
            m["visqol"] = None
            print(f"  [warn] ViSQOL [{i}]: {e}")

        # ── CDPAM: needs [1,1,T] at 22050 Hz ─────────────────────────────────
        if cdpam_model is not None:
            try:
                orig_22k = torchaudio.functional.resample(orig, sample_rate, CDPAM_SR).unsqueeze(0).unsqueeze(0)
                adv_22k  = torchaudio.functional.resample(adv,  sample_rate, CDPAM_SR).unsqueeze(0).unsqueeze(0)
                m["cdpam"] = round(compute_cdpam(cdpam_model, orig_22k, adv_22k), 6)
            except Exception as e:
                m["cdpam"] = None
                print(f"  [warn] CDPAM [{i}]: {e}")
        else:
            m["cdpam"] = None

        # ── PEAQ: needs WAV files at 48 kHz ───────────────────────────────────
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

        all_metrics.append(m)

    return all_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def save_results(log_path: Path, result, overlap: torch.Tensor, args, audio_metrics: list[dict] | None = None) -> None:
    """Persist attack results under log_path/.

    Layout:
        results.json          — per-sample metrics + aggregate + run config
        loss_history.json     — per-step loss values
        audio/
            sample_XX_original.wav
            sample_XX_adversarial.wav
            sample_XX_delta.wav
        heatmaps/
            sample_XX_original.npy
            sample_XX_adversarial.npy
    """
    import torchaudio

    audio_dir = log_path / "audio"
    heatmap_dir = log_path / "heatmaps"
    audio_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    x_orig = (result.x_adv - result.delta).cpu()
    n = result.x_adv.shape[0]

    # ── Audio ────────────────────────────────────────────────────────────────
    for i in range(n):
        torchaudio.save(str(audio_dir / f"sample_{i:02d}_original.wav"),
                        x_orig[i].unsqueeze(0), args.sample_rate)
        torchaudio.save(str(audio_dir / f"sample_{i:02d}_adversarial.wav"),
                        result.x_adv[i].unsqueeze(0).cpu(), args.sample_rate)
        torchaudio.save(str(audio_dir / f"sample_{i:02d}_delta.wav"),
                        result.delta[i].unsqueeze(0).cpu(), args.sample_rate)

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    for i in range(n):
        np.save(str(heatmap_dir / f"sample_{i:02d}_original.npy"),
                result.cam_original[i].cpu().numpy())
        np.save(str(heatmap_dir / f"sample_{i:02d}_adversarial.npy"),
                result.cam_adv[i].cpu().numpy())

    # ── Scalar metrics ────────────────────────────────────────────────────────
    cos = result.cosine_similarity.cpu()
    delta_l2 = result.delta.norm(dim=-1).cpu()

    per_sample = [
        {
            "sample": i,
            "cos_sim": round(cos[i].item(), 6),
            "top10_overlap": round(overlap[i].item(), 6),
            "delta_l2": round(delta_l2[i].item(), 6),
            "pred_preserved": bool(result.prediction_preserved[i].item()),
            **(audio_metrics[i] if audio_metrics else {}),
        }
        for i in range(n)
    ]

    aggregate: dict = {
        "mean_cos_sim": round(cos.mean().item(), 6),
        "mean_top10_overlap": round(overlap.mean().item(), 6),
        "mean_delta_l2": round(delta_l2.mean().item(), 6),
        "delta_linf": round(result.delta.abs().max().item(), 6),
        "pred_preserved_frac": round(
            result.prediction_preserved.float().mean().item(), 6
        ),
    }
    if audio_metrics:
        import numpy as _np
        for key in ("pesq", "stoi", "visqol", "cdpam", "peaq"):
            vals = [m[key] for m in audio_metrics if m.get(key) is not None]
            if vals:
                aggregate[f"mean_{key}"] = round(float(_np.mean(vals)), 6)

    summary = {
        "run": {
            "model_id": args.model_id,
            "audio_dir": str(args.audio_dir),
            "sample_rate": args.sample_rate,
            "clip_seconds": args.clip_seconds,
            "n_samples": args.n_samples,
            "n_steps": args.n_steps,
            "device": args.device,
        },
        "aggregate": aggregate,
        "per_sample": per_sample,
    }

    (log_path / "results.json").write_text(json.dumps(summary, indent=2))
    (log_path / "loss_history.json").write_text(json.dumps(result.history, indent=2))

    print(f"  results.json       → {log_path / 'results.json'}")
    print(f"  loss_history.json  → {log_path / 'loss_history.json'}")
    print(f"  audio/             → {audio_dir}  ({n * 3} files)")
    print(f"  heatmaps/          → {heatmap_dir}  ({n * 2} files)")


def _heatmap_to_image(cam: torch.Tensor) -> torch.Tensor:
    """Normalise a [N_tokens] heatmap to a [1, 1, N_tokens] uint8 image for TensorBoard."""
    cam = cam.float().cpu()
    lo, hi = cam.min(), cam.max()
    cam = (cam - lo) / (hi - lo + 1e-8)          # [0, 1]
    return cam.unsqueeze(0).unsqueeze(0)           # [1, 1, N_tokens]


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace repo ID or local dir, e.g. awsaf49/sonics-SpecTTTra-Sub-2sec",
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory of .wav/.mp3/.flac files; Gaussian noise used when omitted",
    )
    p.add_argument("--sample-rate", type=int, default=16_000,
                   help="Waveform sample rate fed to the model")
    p.add_argument("--clip-seconds", type=float, default=5.0,
                   help="Length of each audio clip")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of waveforms to attack")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Attack optimisation steps")
    p.add_argument("--log-dir", type=Path, default=Path("runs/sonics_attack"),
                   help="TensorBoard log directory")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    print(f"Device: {args.device}")

    # 0 ── Initialise CDPAM (expensive; done once)
    print("\nInitialising CDPAM model…")
    try:
        cdpam_model = _init_cdpam()
        print("  CDPAM ready.")
    except Exception as e:
        cdpam_model = None
        print(f"  [warn] CDPAM init failed ({e}); CDPAM metric will be skipped.")

    # 1 ── Load and wrap the sonics model
    print(f"\nLoading sonics model: {args.model_id}")
    sonics_raw = HFAudioClassifier.from_pretrained(
        args.model_id, map_location=args.device
    )
    model = SonicsWrapper(sonics_raw).to(args.device).eval()
    print(f"  input_shape = {sonics_raw.input_shape}")
    print(f"  n_classes   = {sonics_raw.num_classes}")
    n_tokens = model._m.encoder.st_tokenizer.num_tokens
    n_temporal = model._m.encoder.st_tokenizer.num_temporal_tokens
    n_spectral = model._m.encoder.st_tokenizer.num_spectral_tokens
    print(f"  N_tokens    = {n_tokens} ({n_temporal} temporal + {n_spectral} spectral)")

    # 2 ── Load audio (or synthetic noise)
    x = load_waveforms(
        args.audio_dir,
        args.n_samples,
        args.sample_rate,
        args.clip_seconds,
        args.device,
    )
    print(f"\nInput batch: {list(x.shape)}  mean|x| = {x.abs().mean():.4f}")

    # 3 ── Baseline forward pass
    with torch.no_grad():
        logits = model(x)
    preds = logits.argmax(dim=-1)
    probs = logits.softmax(dim=-1)
    print(f"\nBaseline predictions : {preds.tolist()}")
    print(f"Baseline confidence  : {[f'{v:.3f}' for v in probs.max(dim=-1).values.tolist()]}")

    # 4 ── Run perceptual XAI attack
    gradcam = SpecTTTraGradCAM(model)
    cfg = AttackConfig(n_steps=args.n_steps, log_every=1)   # log every step for TB
    print(f"\nRunning perceptual attack ({cfg.n_steps} steps, lr={cfg.lr})…")
    result = perceptual_xai_attack(model, x, cfg, gradcam=gradcam)

    # 5 ── Console results
    print("\n── Results ─────────────────────────────────────────────────────────")
    cos = result.cosine_similarity
    print(f"Cosine sim (orig↔adv CAM):  {[f'{v:.3f}' for v in cos.tolist()]}")
    print(f"  mean = {cos.mean():.4f}  (lower = explanation changed more)")
    print(f"Prediction preserved:  {result.prediction_preserved.tolist()}")
    print(f"  {result.prediction_preserved.sum().item()}/{args.n_samples} samples kept same class")
    print(f"δ L∞:  {result.delta.abs().max().item():.5f}")
    print(f"δ L2:  {[f'{v:.4f}' for v in result.delta.norm(dim=-1).tolist()]}")
    overlap = topk_overlap(result.cam_original, result.cam_adv, k_frac=0.1)
    print(f"Top-10% token overlap: {[f'{v:.3f}' for v in overlap.tolist()]}")
    print(f"  mean = {overlap.mean():.4f}  (lower = more explanation disagreement)")

    # 5b ── Perceptual audio quality metrics
    x_orig = result.x_adv - result.delta
    print("\n── Perceptual metrics (orig vs adversarial) ────────────────────────")
    audio_metrics = compute_audio_metrics(x_orig, result.x_adv, args.sample_rate, cdpam_model)
    for key in ("pesq", "stoi", "visqol", "cdpam", "peaq"):
        vals = [m[key] for m in audio_metrics if m.get(key) is not None]
        if vals:
            print(f"{key.upper():8s}: {[f'{v:.4f}' for v in vals]}  mean={np.mean(vals):.4f}")

    # 6 ── TensorBoard
    run_name = f"{args.model_id.replace('/', '_')}_{int(time.time())}"
    log_path = args.log_dir / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"\nWriting TensorBoard logs → {log_path}")

    # 6a  Loss curves (one scalar each, plotted against step)
    for entry in result.history:
        s = entry["step"]
        writer.add_scalar("loss/total",       entry["loss"],             s)
        writer.add_scalar("loss/explain",     entry["loss_explain"],     s)
        writer.add_scalar("loss/audibility",  entry["loss_audibility"],  s)
        writer.add_scalar("loss/pred",        entry["loss_pred"],        s)
        writer.add_scalar("attack/cos_sim",   entry["cos_sim"],          s)

    # 6b  Final per-sample metrics (indexed by sample number)
    for i in range(args.n_samples):
        writer.add_scalar("final/cos_sim",          cos[i].item(),                          i)
        writer.add_scalar("final/top10_overlap",    overlap[i].item(),                      i)
        writer.add_scalar("final/delta_l2",         result.delta[i].norm().item(),          i)
        writer.add_scalar("final/pred_preserved",   float(result.prediction_preserved[i]),  i)
        for key in ("pesq", "stoi", "visqol", "cdpam", "peaq"):
            val = audio_metrics[i].get(key)
            if val is not None:
                writer.add_scalar(f"final/{key}", val, i)

    writer.add_scalar("final/mean_cos_sim",       cos.mean().item(),     0)
    writer.add_scalar("final/mean_top10_overlap", overlap.mean().item(), 0)
    writer.add_scalar("final/delta_linf",         result.delta.abs().max().item(), 0)
    writer.add_scalar("final/pred_preserved_frac",
                      result.prediction_preserved.float().mean().item(), 0)
    for key in ("pesq", "stoi", "visqol", "cdpam", "peaq"):
        vals = [audio_metrics[i][key] for i in range(args.n_samples) if audio_metrics[i].get(key) is not None]
        if vals:
            writer.add_scalar(f"final/mean_{key}", float(np.mean(vals)), 0)

    # 6c  Audio: original and adversarial waveforms
    for i in range(args.n_samples):
        writer.add_audio(
            f"audio/sample_{i:02d}_original",
            result.x_adv[i].cpu() - result.delta[i].cpu(),  # reconstruct original
            sample_rate=args.sample_rate,
        )
        writer.add_audio(
            f"audio/sample_{i:02d}_adversarial",
            result.x_adv[i].cpu(),
            sample_rate=args.sample_rate,
        )

    # 6d  Heatmaps: Grad-CAM token importance strips [1, 1, N_tokens]
    for i in range(args.n_samples):
        writer.add_image(
            f"gradcam/sample_{i:02d}_original",
            _heatmap_to_image(result.cam_original[i]),
        )
        writer.add_image(
            f"gradcam/sample_{i:02d}_adversarial",
            _heatmap_to_image(result.cam_adv[i]),
        )

    # 6e  Delta distribution histogram
    writer.add_histogram("delta/values", result.delta.cpu().flatten())

    writer.close()
    print(f"TensorBoard logs written.  Run:  tensorboard --logdir {args.log_dir}")

    # 7 ── Save results to disk
    print(f"\nSaving results → {log_path}")
    save_results(log_path, result, overlap, args, audio_metrics)

    print("\nAttack loop completed successfully.")


if __name__ == "__main__":
    main()