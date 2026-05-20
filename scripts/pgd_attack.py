"""PGD-based XAI attack for AST, VGGish, and Sonics.

Runs the PGD XAI attack using input-gradient saliency maps instead of Grad-CAM.
The perturbation δ is constrained to an L∞ ball and optimised via signed projected
gradient descent to maximise saliency divergence while preserving predictions.

Usage:
    python scripts/pgd_attack.py --model-type ast \\
        --checkpoint runs/ast/version_5/checkpoints/epoch=4-step=9740.ckpt \\
        --data-root audio_xai/data/external \\
        --n-attack-samples 10 --num-iter 100

    python scripts/pgd_attack.py --model-type vggish \\
        --checkpoint runs/vggish/version_3/checkpoints/epoch-epoch=009.ckpt \\
        --data-root audio_xai/data/external \\
        --n-attack-samples 10

    python scripts/pgd_attack.py --model-type sonics \\
        --model-id awsaf49/sonics-spectttra-gamma-5s \\
        --data-root audio_xai/data/external \\
        --n-attack-samples 10

    # Using a YAML config (reads [model], [predict.split] sections)
    python scripts/pgd_attack.py --config config/predict_ast.yaml \\
        --data-root audio_xai/data/external
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

# Shared infrastructure from attack.py (model loading, data, metrics, visualisation)
from attack import (
    _SonicsWrapper,                   # noqa: F401 (imported for load_model to work)
    _strip_lightning_prefix,          # noqa: F401
    load_model,
    load_split,
    load_waveform,
    load_full_waveform,
    _select_balanced_batch,
    AudioPathDataset,
    predict_batch,
    compute_metrics,
    _print_metrics,
    _make_spectrogram_image,
    _metrics_bar_figure,
    _save_sample_images,
    _psychoacoustic_one,
    _psychoacoustic_bar_figure,
    _split_into_windows,
    _stitch_delta,
)

from audio_xai.attacks.pgd_xai_attack import (
    PGDAttackConfig,
    PGDAttackResult,
    pgd_xai_attack,
)
from audio_xai.attacks.perceptual_xai_attack import topk_overlap


# ── PGD micro-batching ────────────────────────────────────────────────────────

def _run_pgd_chunked(
    attack_model,
    x_atk: torch.Tensor,
    cfg: PGDAttackConfig,
    micro_bs: int,
    device: str | None = None,
) -> PGDAttackResult:
    """Run the PGD XAI attack in GPU micro-batches and concatenate results."""
    n = x_atk.shape[0]
    chunks = [x_atk[i : i + micro_bs] for i in range(0, n, micro_bs)]

    parts_x_adv, parts_delta, parts_sal_orig, parts_sal_adv = [], [], [], []
    parts_cos_sim, parts_pred_pres = [], []
    history_chunks: list[list[dict]] = []

    for idx, chunk in enumerate(chunks):
        from tqdm import tqdm as _tqdm
        _tqdm.write(f"  micro-batch {idx + 1}/{len(chunks)}  ({chunk.shape[0]} samples)")
        if device is not None:
            chunk = chunk.to(device)
        result = pgd_xai_attack(attack_model, chunk, cfg)
        parts_x_adv.append(result.x_adv.detach().cpu())
        parts_delta.append(result.delta.detach().cpu())
        parts_sal_orig.append(result.sal_original.detach().cpu())
        parts_sal_adv.append(result.sal_adv.detach().cpu())
        parts_cos_sim.append(result.cosine_similarity.detach().cpu())
        parts_pred_pres.append(result.prediction_preserved.detach().cpu())
        history_chunks.append(result.history)
        del result, chunk
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

    return PGDAttackResult(
        x_adv=torch.cat(parts_x_adv),
        delta=torch.cat(parts_delta),
        sal_original=torch.cat(parts_sal_orig),
        sal_adv=torch.cat(parts_sal_adv),
        cosine_similarity=torch.cat(parts_cos_sim),
        prediction_preserved=torch.cat(parts_pred_pres),
        history=merged_history,
    )


# ── Sliding-window full-audio PGD ─────────────────────────────────────────────

def _attack_full_audio_pgd(
    attack_model,
    x_full: torch.Tensor,
    cfg: PGDAttackConfig,
    clip_len: int,
    hop_len: int,
    micro_bs: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, PGDAttackResult]:
    """Attack a single full-length waveform via sliding-window PGD."""
    windows, T = _split_into_windows(x_full, clip_len, hop_len)
    n_win = windows.shape[0]
    overlap_pct = (1.0 - hop_len / clip_len) * 100.0
    print(f"  sliding-window: {T / cfg.sample_rate:.1f}s → {n_win} windows  "
          f"(clip={clip_len / cfg.sample_rate:.1f}s, "
          f"hop={hop_len / cfg.sample_rate:.1f}s, overlap={overlap_pct:.0f}%)")

    window_result = _run_pgd_chunked(attack_model, windows, cfg, micro_bs, device=device)

    delta_full = _stitch_delta(window_result.delta, clip_len, hop_len, T)
    x_adv_full = (x_full.cpu() + delta_full).clamp(-1.0, 1.0)
    return x_adv_full, delta_full, window_result


# ── Lightning attack module ───────────────────────────────────────────────────

class PGDAttackModule(LightningModule):
    """LightningModule wrapper for the per-sample PGD XAI attack."""

    def __init__(
        self,
        args,
        infer_model: torch.nn.Module,
        attack_model: torch.nn.Module,
        sample_rate: int,
        cfg: PGDAttackConfig,
        clip_len: int,
        hop_len: int,
        sw_micro_bs: int,
        log_path: Path,
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

    @torch.enable_grad()
    def predict_step(self, batch, batch_idx: int) -> dict:
        path_strs, labels_t = batch
        path      = Path(path_strs[0])
        label     = int(labels_t[0].item())
        index_var = batch_idx
        dev       = str(self.device)

        _gpu_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"\n{'='*60}")
        print(f"  [index={index_var}] {path.name}  "
              f"(label={'real' if label==0 else 'fake'})  GPU: {_gpu_used:.2f} GiB")
        print(f"{'='*60}")

        max_oom_retries = getattr(self.args, 'oom_retries', 3)
        result: dict = {"index": index_var, "stem": path.stem, "label": label, "ok": False}

        for oom_attempt in range(max_oom_retries):
            if oom_attempt > 0:
                _gpu_now = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(f"\n  [RETRY {oom_attempt}/{max_oom_retries-1}]  GPU: {_gpu_now:.2f} GiB")

            x_single = x_clip = x_adv_clip = None
            x_adv_full = delta_full = delta_clip = None
            orig_cpu = adv_cpu = orig_save = adv_save = dlt_save = None
            pred_orig = prob_orig = pred_adv = prob_adv = None
            cos_sim_t = pred_pres_t = None
            sal_orig_np: np.ndarray | None = None
            sal_adv_np:  np.ndarray | None = None
            history: list[dict] = []

            try:
                if self.args.full_audio:
                    x_single = load_full_waveform(path, self.sample_rate)
                else:
                    x_single = load_waveform(path, self.sample_rate, self.clip_len)

                x_clip = x_single[:self.clip_len].unsqueeze(0).to(dev)
                pred_orig, prob_orig = predict_batch(self.infer_model, x_clip)

                if self.args.full_audio:
                    x_adv_full, delta_full, win_result = _attack_full_audio_pgd(
                        self.attack_model, x_single, self.cfg,
                        self.clip_len, self.hop_len, self.sw_micro_bs, dev,
                    )
                    x_adv_clip  = x_adv_full[:self.clip_len].unsqueeze(0).to(dev)
                    cos_sim_t   = win_result.cosine_similarity
                    pred_pres_t = win_result.prediction_preserved
                    history     = win_result.history
                    _s_o = win_result.sal_original.detach().cpu().numpy()
                    _s_a = win_result.sal_adv.detach().cpu().numpy()
                    # [N, T] → [N*T] (1-D saliency, one vector per window)
                    sal_orig_np = _s_o.reshape(-1)
                    sal_adv_np  = _s_a.reshape(-1)
                    del win_result, _s_o, _s_a
                else:
                    res         = pgd_xai_attack(self.attack_model, x_clip, self.cfg)
                    x_adv_clip  = res.x_adv
                    delta_clip  = res.delta.squeeze(0).detach().cpu()
                    cos_sim_t   = res.cosine_similarity
                    pred_pres_t = res.prediction_preserved
                    history     = res.history
                    sal_orig_np = res.sal_original[0].detach().cpu().numpy()
                    sal_adv_np  = res.sal_adv[0].detach().cpu().numpy()
                    del res

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

                if sal_orig_np is not None:
                    sal_o_t  = torch.from_numpy(sal_orig_np).unsqueeze(0)
                    sal_a_t  = torch.from_numpy(sal_adv_np).unsqueeze(0)
                    over_val = float(topk_overlap(sal_o_t, sal_a_t, k_frac=0.1).item())
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

                if sal_orig_np is not None:
                    hmap_dir = self.log_path / "heatmaps" / path.stem
                    hmap_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(hmap_dir / "saliency_original.npy"),    sal_orig_np)
                    np.save(str(hmap_dir / "saliency_adversarial.npy"), sal_adv_np)
                    _save_sample_images(
                        sample_dir  = self.log_path / "images" / path.stem,
                        orig_wav    = orig_save if self.args.full_audio else orig_cpu,
                        adv_wav     = adv_save  if self.args.full_audio else adv_cpu,
                        delta_wav   = dlt_save,
                        cam_orig    = sal_orig_np,
                        cam_adv     = sal_adv_np,
                        sample_rate = self.sample_rate,
                        stem        = path.stem,
                        label       = label,
                        pred_orig   = int(pred_orig.item()),
                        pred_adv    = int(pred_adv.item()),
                    )

                result = {**sample_metrics, "history": history}
                break

            except torch.cuda.OutOfMemoryError as oom_e:
                print(f"\n  [OOM] index={index_var}, "
                      f"attempt {oom_attempt + 1}/{max_oom_retries}: {oom_e}")

            except Exception as exc:
                import traceback as _tb
                print(f"  [ERROR] index={index_var} ({path.name}): {exc}")
                _tb.print_exc()
                break

            finally:
                for _v in (x_single, x_clip, x_adv_clip, x_adv_full,
                           delta_full, delta_clip, orig_cpu, adv_cpu,
                           orig_save, adv_save, dlt_save,
                           pred_orig, prob_orig, pred_adv, prob_adv,
                           cos_sim_t, pred_pres_t):
                    _v = None  # noqa: F841
                sal_orig_np = sal_adv_np = None
                self.attack_model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            if oom_attempt < max_oom_retries - 1:
                _wait = 2 * (oom_attempt + 1)
                print(f"  Waiting {_wait}s before retry...")
                time.sleep(_wait)
            else:
                print(f"  [OOM] All {max_oom_retries} attempts exhausted — skipping")

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
                   help="YAML config file (reads [model], [predict.split] sections)")

    # Model
    p.add_argument("--model-type", required=False, choices=["sonics", "ast", "vggish"])
    p.add_argument("--model-id",   default="awsaf49/sonics-spectttra-gamma-5s")
    p.add_argument("--checkpoint", type=Path, default=None)

    # Data
    p.add_argument("--data-root",    type=Path, required=False)
    p.add_argument("--split",        default="test")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate",  type=int, default=16_000)
    p.add_argument("--seed",         type=int, default=42)

    # PGD attack hyperparameters
    p.add_argument("--n-attack-samples",  type=int,   default=None)
    p.add_argument("--attack-micro-batch",type=int,   default=None)
    p.add_argument("--eps",               type=float, default=0.01,
                   help="L∞ radius of the perturbation ball")
    p.add_argument("--alpha",             type=float, default=0.001,
                   help="Signed-gradient step size per PGD iteration")
    p.add_argument("--num-iter",          type=int,   default=100,
                   help="Number of PGD iterations")
    p.add_argument("--lambda-aud",        type=float, default=0.5)
    p.add_argument("--lambda-pred",       type=float, default=100.0)
    p.add_argument("--oom-retries",       type=int,   default=3)

    # Sliding-window full-audio mode (same as attack.py)
    p.add_argument("--full-audio",         action="store_true", default=False)
    p.add_argument("--window-hop-seconds", type=float, default=None)

    # Batching
    p.add_argument("--n-batches",   type=int, default=1)
    p.add_argument("--batch-index", type=int, default=0)

    # Output
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-dir",  type=Path, default=Path("runs/pgd_attack"))
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")

    if _pre_args.config:
        from audio_xai.hparams import pgd_attack_defaults
        p.set_defaults(**pgd_attack_defaults(_pre_args.config))

    args = p.parse_args()

    if args.model_type is None:
        p.error("--model-type is required")
    if args.data_root is None:
        p.error("--data-root is required")
    if args.n_attack_samples is None:
        p.error("--n-attack-samples is required")

    print(f"Device : {args.device}")
    print(f"Model  : {args.model_type}")
    print(f"PGD    : eps={args.eps}  alpha={args.alpha}  num_iter={args.num_iter}")

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

    # ── 3. Select balanced sample list ────────────────────────────────────────
    print(f"\n── PGD XAI attack  ({args.n_attack_samples} samples, {args.num_iter} iterations) ──")
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

    print(f"Selected {len(atk_paths)} samples")

    # ── 4. Setup logging ──────────────────────────────────────────────────────
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else args.model_id.replace("/", "_")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = args.run_name or f"pgd_{args.model_type}_{ckpt_tag}_{args.split}_{ts}"

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
        atk_paths, gt_labels_list = (map(list, zip(*filtered)) if filtered else ([], []))
        skipped = before - len(atk_paths)
        print(f"\nResume: {skipped}/{before} already done, {len(atk_paths)} remaining")
        for stem in sorted(done_stems):
            print(f"  [done]  {stem}")
        for p in atk_paths:
            print(f"  [todo]  {Path(p).stem}")
    else:
        print(f"\nResume: no prior output — processing all {len(atk_paths)} samples")

    if not atk_paths:
        print("All samples already processed — nothing to do.")
        return

    # ── 5. Attack config ───────────────────────────────────────────────────────
    cfg = PGDAttackConfig(
        eps=args.eps,
        alpha=args.alpha,
        num_iter=args.num_iter,
        lambda_aud=args.lambda_aud,
        lambda_pred=args.lambda_pred,
        log_every=1,
        sample_rate=sample_rate,
    )
    hop_len = int((args.window_hop_seconds or args.clip_seconds) * sample_rate)
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
    module = PGDAttackModule(
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
        inference_mode       = False,
        plugins              = plugins or None,
    )

    # ── 9. Run predict ─────────────────────────────────────────────────────────
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

    print(f"\n  Saliency cosine sim : mean={mean_cos:.4f}  (lower = more change)")
    print(f"  Top-10% overlap     : mean={mean_over:.4f}  (lower = more disagreement)")
    print(f"  δ L∞ mean           : {mean_linf:.5f}")
    print(f"  Pred preserved      : {sum(all_pred_pres)}/{n_done}")

    print(f"\n── Δ (adversarial − original) ──────────────────────────────────────")
    for k in ("accuracy", "f1", "auc"):
        ov, av = orig_atk_metrics[k], adv_metrics[k]
        print(f"  Δ{k:10s}: {av - ov:+.4f}  ({ov:.4f} → {av:.4f})")

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

    summary: dict = {
        "run": {
            "attack":             "pgd",
            "model_type":         args.model_type,
            "model_id":           args.model_id if args.model_type == "sonics" else None,
            "checkpoint":         str(args.checkpoint) if args.checkpoint else None,
            "split":              args.split,
            "clip_seconds":       args.clip_seconds,
            "sample_rate":        sample_rate,
            "n_attack_samples":   n_done,
            "eps":                args.eps,
            "alpha":              args.alpha,
            "num_iter":           args.num_iter,
            "lambda_aud":         args.lambda_aud,
            "lambda_pred":        args.lambda_pred,
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
    writer = SummaryWriter(log_dir=str(log_path))

    def _metrics_md_table(metrics: dict, title: str) -> str:
        rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items()
                         if isinstance(v, float))
        return f"**{title}**\n\n| Metric | Value |\n|--------|-------|\n{rows}"

    fig = _metrics_bar_figure(
        orig_atk_metrics, adv_metrics,
        title=f"PGD {args.model_type} — Classification metrics ({n_done} samples)",
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
                    _metrics_md_table(atk_summary, "PGD attack summary"), 0)
    if perceptual_means:
        writer.add_text("metrics/perceptual",
                        _metrics_md_table(perceptual_means, "Perceptual quality"), 0)

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
            title=f"PGD {args.model_type} — Perceptual Quality ({n_done} samples)",
        )
        writer.add_figure("attack/metrics_comparison_psychoacoustic", fig, 0)
        plt.close(fig)

    writer.close()
    print(f"\nTensorBoard → {log_path}")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()