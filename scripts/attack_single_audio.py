"""Single-audio adversarial attack demo with GradCAM + LRP explanation and STOI.

Randomly picks one audio clip (real or fake) from the dataset, runs three
attacks against it, then shows GradCAM and LRP explanations before/after
each attack and reports STOI for every perturbed output.

Attacks used
------------
1. Perceptual XAI attack  (Adam, psychoacoustic masking, GradCAM-based)
2. PGD XAI attack         (signed gradient descent, input-gradient saliency)
3. X-Shift attack         (sparse shift toward reversed saliency target)

Usage
-----
    python scripts/attack_single_audio.py \
        --model-type vggish \
        --checkpoint runs/vggish/version_3/checkpoints/epoch-epoch=009.ckpt \
        --data-root audio_xai/data/external \
        --clip-seconds 5 \
        --seed 42 \
        --out-dir runs/attack_single_demo
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pystoi import stoi

from audio_xai.attacks.perceptual_xai_attack import (
    AttackConfig,
    perceptual_xai_attack,
)
from audio_xai.attacks.pgd_xai_attack import PGDAttackConfig, pgd_xai_attack
from audio_xai.attacks.xshift_attack import XShiftConfig, xshift_attack
from audio_xai.models.ast_binary import ASTBinary, AST_SAMPLE_RATE
from audio_xai.models.vggish_binary import VGGishBinary, VGGISH_SAMPLE_RATE
from audio_xai.models.wav2vec2_binary import Wav2Vec2Binary, WAV2VEC2_SAMPLE_RATE
from audio_xai.xai.gradcam import make_gradcam
from audio_xai.xai.lrp import make_lrp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_lightning_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("_"):
            continue
        out[k[6:] if k.startswith("model.") else k] = v
    return out


def load_model(args) -> tuple[torch.nn.Module, int]:
    device = args.device

    def _load_ckpt(model):
        if not args.checkpoint:
            return
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = _strip_lightning_prefix(ck.get("state_dict", ck))
        model_sd = model.state_dict()
        bad = [k for k, v in sd.items()
               if k in model_sd and v.shape != model_sd[k].shape]
        if bad:
            sd = {k: v for k, v in sd.items() if k not in bad}
        model.load_state_dict(sd, strict=False)

    if args.model_type == "ast":
        model = ASTBinary(pretrained=not args.checkpoint, clip_seconds=args.clip_seconds)
        _load_ckpt(model)
        return model.to(device).eval(), AST_SAMPLE_RATE

    if args.model_type == "vggish":
        model = VGGishBinary()
        _load_ckpt(model)
        return model.to(device).eval(), VGGISH_SAMPLE_RATE

    if args.model_type == "wav2vec2":
        model = Wav2Vec2Binary(pretrained=not args.checkpoint)
        _load_ckpt(model)
        return model.to(device).eval(), WAV2VEC2_SAMPLE_RATE

    raise ValueError(f"Unknown model-type: {args.model_type}")


def pick_random_audio(data_root: Path, seed: int) -> tuple[Path, int]:
    """Return (audio_path, label) — label 0=real, 1=fake."""
    rng = random.Random(seed)
    candidates: list[tuple[Path, int]] = []
    for label, folder_name in ((0, "real_songs"), (1, "fake_songs")):
        folder = data_root / folder_name
        if not folder.is_dir():
            # fall back to real/fake subdirectory names
            folder = data_root / ("real" if label == 0 else "fake")
        if folder.is_dir():
            for ext in (".wav", ".mp3", ".flac"):
                candidates.extend((p, label) for p in sorted(folder.glob(f"*{ext}")))
    if not candidates:
        raise FileNotFoundError(f"No audio files found under {data_root}")
    return rng.choice(candidates)


def load_waveform(path: Path, sample_rate: int, clip_len: int) -> torch.Tensor:
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]
    return torch.from_numpy(y)


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """STOI in [0, 1]; clip to avoid numerical edge cases."""
    try:
        s = stoi(ref, deg, sr, extended=False)
        return float(np.clip(s, 0.0, 1.0))
    except Exception as e:
        print(f"  [warn] STOI failed: {e}")
        return float("nan")


# ── Spectrogram / heatmap utilities ──────────────────────────────────────────

def _mel_spec_db(wav: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=wav, sr=sr,
                                         n_fft=1024, hop_length=256, n_mels=n_mels)
    db  = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    lo, hi = db.min(), db.max()
    return ((db - lo) / (hi - lo + 1e-8))[::-1]  # high freq at top, [0,1]


def _to_rgb(arr: np.ndarray, cmap: str) -> np.ndarray:
    """Normalised 2-D array → [H, W, 3] float32 via a matplotlib colormap."""
    lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / (hi - lo + 1e-8)
    return plt.get_cmap(cmap)(norm)[:, :, :3].astype(np.float32)


def _resize_hw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def _to_2d(heatmap: np.ndarray) -> np.ndarray:
    """Collapse any heatmap shape to 2D [H, W].

    VGGish splits a 5s clip into ~6 patches of 0.96s, so GradCAM returns
    [N_patches, H, W].  PGD/XShift return a 1-D saliency [T].  Both cases
    need to reach a 2D array before bilinear resize.
    """
    h = np.asarray(heatmap)
    if h.ndim == 1:
        return h[np.newaxis, :]          # [1, T]
    while h.ndim > 2:
        h = h.mean(axis=0)               # average over leading patch/batch dim
    return h


def spec_and_heatmap(
    wav: torch.Tensor,
    heatmap: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (spec_rgb, hmap_rgb, overlay) each [H, W, 3]."""
    spec_raw = _mel_spec_db(wav.cpu().numpy(), sr)
    H, W     = spec_raw.shape
    spec_rgb = _to_rgb(spec_raw, "inferno")
    hmap_rs  = _resize_hw(_to_2d(heatmap), H, W)
    hmap_rgb = _to_rgb(hmap_rs, "hot")
    overlay  = np.clip(0.55 * spec_rgb + 0.45 * hmap_rgb, 0, 1)
    return spec_rgb, hmap_rgb, overlay


# ── GradCAM on any waveform (no grad needed for display) ────────────────────

def compute_gradcam(model, wav_1d: torch.Tensor, device: str) -> np.ndarray:
    x = wav_1d.unsqueeze(0).to(device)
    with make_gradcam(model) as gc:
        hmap = gc(x, create_graph=False)   # [1, H, W] or [1, T]
    return hmap.squeeze(0).detach().cpu().numpy()


# ── LRP on any waveform ───────────────────────────────────────────────────────

def compute_lrp(model, wav_1d: torch.Tensor, device: str) -> np.ndarray | None:
    """Returns 2-D heatmap or None if model is not LRP-supported (e.g. Sonics)."""
    try:
        x = wav_1d.unsqueeze(0).to(device)
        with make_lrp(model) as lrp_fn:
            with torch.enable_grad():
                hmap = lrp_fn(x)           # [1, H, W]
        return hmap.squeeze(0).detach().cpu().numpy()
    except Exception as e:
        print(f"  [warn] LRP skipped: {e}")
        return None


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_xai_panel(
    wav_orig: torch.Tensor,
    wav_adv: torch.Tensor,
    heatmap_orig: np.ndarray,
    heatmap_adv: np.ndarray,
    sr: int,
    method_name: str,
    attack_name: str,
    label: int,
    pred_orig: int,
    pred_adv: int,
    stoi_val: float,
    cos_sim: float,
    save_path: Path,
) -> None:
    """6-panel figure: original (spec / hmap / overlay) | adversarial (spec / hmap / overlay)."""
    spec_o, hmap_o, over_o = spec_and_heatmap(wav_orig, heatmap_orig, sr)
    spec_a, hmap_a, over_a = spec_and_heatmap(wav_adv,  heatmap_adv,  sr)

    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.05)

    titles_top = ["Spectrogram (original)", f"{method_name} heatmap (original)", "Overlay (original)"]
    titles_bot = ["Spectrogram (adversarial)", f"{method_name} heatmap (adversarial)", "Overlay (adversarial)"]
    imgs_top = [spec_o, hmap_o, over_o]
    imgs_bot = [spec_a, hmap_a, over_a]

    for col, (img, ttl) in enumerate(zip(imgs_top, titles_top)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, aspect="auto", origin="upper")
        ax.set_title(ttl, fontsize=10, pad=4)
        ax.axis("off")

    for col, (img, ttl) in enumerate(zip(imgs_bot, titles_bot)):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img, aspect="auto", origin="upper")
        ax.set_title(ttl, fontsize=10, pad=4)
        ax.axis("off")

    true_lbl = "real" if label == 0 else "fake"
    fig.suptitle(
        f"{attack_name} | {method_name}  "
        f"true={true_lbl}  pred_orig={'real' if pred_orig==0 else 'fake'}  "
        f"pred_adv={'real' if pred_adv==0 else 'fake'}  "
        f"cos_sim={cos_sim:.3f}  STOI={stoi_val:.3f}",
        fontsize=10,
    )
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


def _plot_waveform_diff(
    wav_orig: torch.Tensor,
    wav_adv: torch.Tensor,
    sr: int,
    attack_name: str,
    save_path: Path,
) -> None:
    o = wav_orig.cpu().numpy()
    a = wav_adv.cpu().numpy()
    d = a - o

    t = np.arange(len(o)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t, o, lw=0.4, color="steelblue")
    axes[0].set_ylabel("Original", fontsize=9)
    axes[1].plot(t, a, lw=0.4, color="darkorange")
    axes[1].set_ylabel("Adversarial", fontsize=9)
    axes[2].plot(t, d, lw=0.4, color="crimson")
    axes[2].set_ylabel("Perturbation δ", fontsize=9)
    axes[2].set_xlabel("Time (s)", fontsize=9)

    linf = np.abs(d).max()
    l2   = np.sqrt((d ** 2).mean())
    fig.suptitle(f"{attack_name} — waveform diff  L∞={linf:.5f}  L2={l2:.5f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-type",   default="vggish", choices=["vggish", "ast", "wav2vec2"])
    p.add_argument("--checkpoint",   type=Path, default=None)
    p.add_argument("--data-root",    type=Path,
                   default=Path("audio_xai/data/external"))
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--out-dir",      type=Path, default=Path("runs/attack_single_demo"))
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    # Attack hyper-params (keep defaults conservative for a 5s demo)
    p.add_argument("--n-steps",      type=int, default=100,
                   help="Steps for Perceptual XAI attack (Adam)")
    p.add_argument("--pgd-iter",     type=int, default=100,
                   help="Iterations for PGD attack")
    p.add_argument("--xshift-iter",  type=int, default=100,
                   help="Iterations for X-Shift attack")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Device      : {args.device}")
    print(f"Model       : {args.model_type}")
    print(f"Checkpoint  : {args.checkpoint or '(pretrained)'}")
    print(f"Clip length : {args.clip_seconds}s")
    print(f"Seed        : {args.seed}")
    print(f"Output      : {args.out_dir}")
    print(f"{'='*60}\n")

    # ── 1. Pick a random audio sample ─────────────────────────────────────────
    audio_path, label = pick_random_audio(args.data_root, args.seed)
    true_label_str = "REAL" if label == 0 else "FAKE"
    print(f"Selected    : {audio_path.name}  [{true_label_str}]")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model, sr = load_model(args)
    clip_len = int(args.clip_seconds * sr)
    print(f"Sample rate : {sr} Hz  |  clip samples: {clip_len}")

    # ── 3. Load waveform ──────────────────────────────────────────────────────
    wav = load_waveform(audio_path, sr, clip_len)        # [T]
    x   = wav.unsqueeze(0).to(args.device)               # [1, T]

    with torch.no_grad():
        logits_orig = model(x)
        if logits_orig.shape[1] == 1:
            pred_orig = int((logits_orig.squeeze(1) >= 0).long().item())
        else:
            pred_orig = int(logits_orig.argmax(dim=-1).item())
    print(f"Prediction  : {'real' if pred_orig==0 else 'fake'}  "
          f"(true: {true_label_str.lower()})")

    # ── 4. Pre-attack XAI baselines ───────────────────────────────────────────
    print("\n[Baseline] Computing GradCAM and LRP on clean audio…")
    gc_orig  = compute_gradcam(model, wav, args.device)
    lrp_orig = compute_lrp(model, wav, args.device)

    # Save clean audio
    torchaudio.save(str(args.out_dir / "original.wav"), wav.unsqueeze(0), sr)

    # ── 5. Define three attacks ───────────────────────────────────────────────
    attacks = [
        {
            "name": "1_Perceptual_XAI",
            "label": "Perceptual XAI (Adam)",
            "fn": lambda: perceptual_xai_attack(
                model, x,
                cfg=AttackConfig(
                    n_steps=args.n_steps,
                    sample_rate=sr,
                    log_every=None,
                ),
            ),
            "get_adv":   lambda res: res.x_adv.squeeze(0).cpu(),
            "get_cam_a": lambda res: res.cam_adv.squeeze(0).cpu().numpy(),
            "get_cos":   lambda res: float(res.cosine_similarity.mean().item()),
            "get_pred":  lambda res: int((model(res.x_adv).argmax(-1) if model(res.x_adv).shape[1]>1
                                          else (model(res.x_adv).squeeze(1)>=0).long()).item()),
        },
        {
            "name": "2_PGD_XAI",
            "label": "PGD XAI (input-gradient saliency)",
            "fn": lambda: pgd_xai_attack(
                model, x,
                cfg=PGDAttackConfig(
                    num_iter=args.pgd_iter,
                    sample_rate=sr,
                    log_every=None,
                ),
            ),
            "get_adv":   lambda res: res.x_adv.squeeze(0).cpu(),
            "get_cam_a": lambda res: res.sal_adv.squeeze(0).cpu().numpy(),
            "get_cos":   lambda res: float(res.cosine_similarity.mean().item()),
            "get_pred":  lambda res: int((model(res.x_adv).argmax(-1) if model(res.x_adv).shape[1]>1
                                          else (model(res.x_adv).squeeze(1)>=0).long()).item()),
        },
        {
            "name": "3_XShift",
            "label": "X-Shift (reverse-target, sparse)",
            "fn": lambda: xshift_attack(
                model, x,
                cfg=XShiftConfig(
                    num_iter=args.xshift_iter,
                    target_mode="reverse",
                    sample_rate=sr,
                    log_every=None,
                ),
            ),
            "get_adv":   lambda res: res.x_adv.squeeze(0).cpu(),
            "get_cam_a": lambda res: res.sal_adv.squeeze(0).cpu().numpy(),
            "get_cos":   lambda res: float(res.cos_orig_adv.mean().item()),
            "get_pred":  lambda res: int((model(res.x_adv).argmax(-1) if model(res.x_adv).shape[1]>1
                                          else (model(res.x_adv).squeeze(1)>=0).long()).item()),
        },
    ]

    # ── 6. Run each attack, save outputs, compute STOI + XAI explanations ────
    ref_np = wav.cpu().numpy()
    summary: list[dict] = []

    for atk in attacks:
        name  = atk["name"]
        label_str = atk["label"]
        print(f"\n{'─'*55}")
        print(f"Attack: {label_str}")
        print(f"{'─'*55}")

        # Run attack
        with torch.no_grad():
            pass  # ensure model is in no-grad mode except inside attack
        result = atk["fn"]()

        wav_adv  = atk["get_adv"](result)     # [T] cpu
        cos_sim  = atk["get_cos"](result)

        with torch.no_grad():
            if model(wav_adv.unsqueeze(0).to(args.device)).shape[1] > 1:
                pred_adv = int(model(wav_adv.unsqueeze(0).to(args.device)).argmax(-1).item())
            else:
                pred_adv = int((model(wav_adv.unsqueeze(0).to(args.device)).squeeze(1) >= 0).long().item())

        pred_preserved = pred_adv == pred_orig

        # STOI
        deg_np  = wav_adv.numpy()
        stoi_val = compute_stoi(ref_np, deg_np, sr)

        # Delta stats
        delta    = deg_np - ref_np
        linf     = float(np.abs(delta).max())
        l2       = float(np.sqrt((delta ** 2).mean()))
        snr      = 10 * np.log10((ref_np ** 2).mean() / ((delta ** 2).mean() + 1e-12))

        print(f"  L∞={linf:.5f}  L2={l2:.5f}  SNR={snr:.1f}dB")
        print(f"  pred_orig={'real' if pred_orig==0 else 'fake'}  "
              f"pred_adv={'real' if pred_adv==0 else 'fake'}  "
              f"preserved={pred_preserved}")
        print(f"  Cosine similarity (XAI): {cos_sim:.4f}")
        print(f"  STOI                   : {stoi_val:.4f}")

        # Save adversarial audio
        adv_wav_path = args.out_dir / f"{name}_adversarial.wav"
        torchaudio.save(str(adv_wav_path), wav_adv.unsqueeze(0), sr)

        # ── GradCAM before / after ────────────────────────────────────────────
        print("  Computing GradCAM on adversarial audio…")
        gc_adv = compute_gradcam(model, wav_adv, args.device)

        _plot_xai_panel(
            wav_orig=wav,
            wav_adv=wav_adv,
            heatmap_orig=gc_orig,
            heatmap_adv=gc_adv,
            sr=sr,
            method_name="GradCAM",
            attack_name=label_str,
            label=label,
            pred_orig=pred_orig,
            pred_adv=pred_adv,
            stoi_val=stoi_val,
            cos_sim=cos_sim,
            save_path=args.out_dir / f"{name}_gradcam.png",
        )

        # ── LRP before / after ────────────────────────────────────────────────
        if lrp_orig is not None:
            print("  Computing LRP on adversarial audio…")
            lrp_adv = compute_lrp(model, wav_adv, args.device)
            if lrp_adv is not None:
                lrp_cos = float(F.cosine_similarity(
                    torch.from_numpy(lrp_orig).flatten().unsqueeze(0),
                    torch.from_numpy(lrp_adv).flatten().unsqueeze(0),
                ).item())
                print(f"  LRP cosine similarity: {lrp_cos:.4f}")
                _plot_xai_panel(
                    wav_orig=wav,
                    wav_adv=wav_adv,
                    heatmap_orig=lrp_orig,
                    heatmap_adv=lrp_adv,
                    sr=sr,
                    method_name="LRP",
                    attack_name=label_str,
                    label=label,
                    pred_orig=pred_orig,
                    pred_adv=pred_adv,
                    stoi_val=stoi_val,
                    cos_sim=lrp_cos,
                    save_path=args.out_dir / f"{name}_lrp.png",
                )
            else:
                lrp_cos = float("nan")
        else:
            lrp_cos = float("nan")

        # ── Waveform diff ─────────────────────────────────────────────────────
        _plot_waveform_diff(
            wav_orig=wav,
            wav_adv=wav_adv,
            sr=sr,
            attack_name=label_str,
            save_path=args.out_dir / f"{name}_waveform_diff.png",
        )

        summary.append({
            "attack": label_str,
            "pred_orig": "real" if pred_orig == 0 else "fake",
            "pred_adv": "real" if pred_adv == 0 else "fake",
            "pred_preserved": pred_preserved,
            "STOI": round(stoi_val, 4),
            "cos_sim_gradcam": round(cos_sim, 4),
            "cos_sim_lrp": round(lrp_cos, 4) if not np.isnan(lrp_cos) else "n/a",
            "L_inf": round(linf, 6),
            "L2": round(l2, 6),
            "SNR_dB": round(float(snr), 2),
        })

    # ── 7. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY  |  file: {audio_path.name}  [{true_label_str}]")
    print(f"{'='*60}")
    col_w = 34
    hdr = f"{'Attack':<{col_w}} {'Pred':>6}  {'STOI':>6}  {'GC-cos':>7}  {'LRP-cos':>8}  {'L∞':>8}  {'SNR(dB)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in summary:
        pred_str = f"{row['pred_adv']}{'✓' if row['pred_preserved'] else '✗'}"
        lrp_str = f"{row['cos_sim_lrp']}" if isinstance(row["cos_sim_lrp"], float) else row["cos_sim_lrp"]
        print(
            f"{row['attack']:<{col_w}} {pred_str:>6}  "
            f"{row['STOI']:>6.4f}  {row['cos_sim_gradcam']:>7.4f}  "
            f"{lrp_str:>8}  {row['L_inf']:>8.5f}  {row['SNR_dB']:>8.2f}"
        )
    print()

    # ── 8. Persist summary as JSON ────────────────────────────────────────────
    import json
    out_json = args.out_dir / "summary.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "audio_file": str(audio_path),
                "true_label": true_label_str.lower(),
                "model": args.model_type,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "clip_seconds": args.clip_seconds,
                "attacks": summary,
            },
            f,
            indent=2,
        )
    print(f"JSON summary → {out_json}")
    print(f"All outputs  → {args.out_dir}/")


if __name__ == "__main__":
    main()
