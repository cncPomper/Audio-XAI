"""Sparse Audio Focus-Shift (X-Shift) attack.

Shifts the model's XAI saliency map toward a target explanation using signed
projected gradient descent with L∞ + L0 (TopK sparsity) constraints.

Loss:
    L(δ) = 1 − cos_sim(sal(x+δ), target_sal)   — X-Shift: align to target
          + λ_pred · KL(f(x+δ) ‖ f(x))          — preserve prediction

Update:
    δ ← clamp(δ − α · sign(∇_δ L),  −ε, +ε)    — signed GD + L∞ projection
    δ ← TopK(δ, k)                               — L0 sparsity projection

Target modes:
    "shift"   — roll original saliency by `target_shift_frac * T` samples in time
    "reverse" — invert saliency: high-attention → low-attention regions
    "uniform" — push toward uniform attention across all time steps
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from audio_xai.attacks.pgd_xai_attack import (
    _compute_saliency,
    _flatten_normalize,
)


@dataclass
class XShiftConfig:
    eps: float = 0.01                # L∞ radius of the perturbation ball
    alpha: float = 0.002             # signed-gradient step size per iteration
    num_iter: int = 100              # number of PGD iterations
    sparsity_ratio: float = 0.05    # fraction of T allowed to be non-zero (L0 budget)
    lambda_pred: float = 100.0      # KL prediction-preservation weight
    target_mode: str = "shift"      # "shift" | "reverse" | "uniform"
    target_shift_frac: float = 0.5  # for "shift": roll by this fraction of T
    sample_rate: int = 16_000
    log_every: int | None = 20


@dataclass
class XShiftResult:
    x_adv: torch.Tensor                 # perturbed waveform [B, T]
    delta: torch.Tensor                 # perturbation [B, T]
    sal_original: torch.Tensor          # reference saliency on clean input [B, T]
    sal_adv: torch.Tensor               # saliency on adversarial input [B, T]
    target_sal: torch.Tensor            # target saliency [B, T]
    cos_to_target: torch.Tensor         # cos_sim(sal_adv, target_sal) per sample [B]
    cos_orig_adv: torch.Tensor          # cos_sim(sal_orig, sal_adv) per sample [B]
    prediction_preserved: torch.Tensor  # bool [B]
    history: list[dict] = field(default_factory=list)


def make_target_explanation(
    sal_orig: torch.Tensor,
    mode: str,
    shift_frac: float = 0.5,
) -> torch.Tensor:
    """Derive a target saliency from the original.

    Args:
        sal_orig: Original saliency [B, T], detached, non-negative.
        mode: "shift" | "reverse" | "uniform"
        shift_frac: Fraction of T to roll for mode="shift".
    """
    s = sal_orig.detach()
    T = s.shape[-1]
    if mode == "shift":
        shift = max(1, int(shift_frac * T))
        return torch.roll(s, shifts=shift, dims=-1)
    elif mode == "reverse":
        # Flip: high-salience regions become low-salience targets and vice versa
        s_max = s.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return s_max - s
    elif mode == "uniform":
        return torch.ones_like(s)
    else:
        raise ValueError(f"Unknown target_mode: {mode!r}. Choose shift | reverse | uniform")


def _topk_sparsity(delta: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all elements except the top-k by absolute value per batch row."""
    B = delta.shape[0]
    flat = delta.view(B, -1)
    n = flat.shape[-1]
    if k >= n:
        return delta.clone()
    abs_flat = flat.abs()
    # (n - k + 1)-th smallest = k-th largest
    kth_rank = n - k + 1
    thresholds, _ = torch.kthvalue(abs_flat, kth_rank, dim=-1)  # (B,)
    mask = (abs_flat >= thresholds.unsqueeze(-1)).float()
    return (flat * mask).view_as(delta)


def xshift_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg: XShiftConfig | None = None,
) -> XShiftResult:
    """Run the X-Shift attack on a batch of waveforms.

    Args:
        model: Trained AudioClassifier in eval mode.
        x: Input waveforms [B, T] on the model's device.
        cfg: X-Shift hyperparameters; defaults to XShiftConfig().
    """
    cfg = cfg or XShiftConfig()

    # ── 1. Baseline predictions and original probability distribution ─────
    with torch.no_grad():
        logits_orig = model(x)
        if logits_orig.shape[1] == 1:
            pred_orig = (logits_orig.squeeze(1) >= 0).long()
            p = torch.sigmoid(logits_orig)
            orig_probs = torch.cat([1 - p, p], dim=1)
        else:
            pred_orig = logits_orig.argmax(dim=-1)
            orig_probs = F.softmax(logits_orig, dim=-1)

    # ── 2. Reference saliency on clean input ─────────────────────────────
    x_clean = x.detach().requires_grad_(True)
    with torch.enable_grad():
        _, sal_orig = _compute_saliency(model, x_clean, pred_orig, create_graph=False)
    sal_orig = sal_orig.detach()
    del x_clean

    sal_orig_flat = _flatten_normalize(sal_orig)

    # ── 3. Derive target saliency ─────────────────────────────────────────
    target_sal = make_target_explanation(sal_orig, cfg.target_mode, cfg.target_shift_frac)
    target_sal_flat = _flatten_normalize(target_sal)

    # ── 4. L0 budget: k non-zero elements per sample ─────────────────────
    T = x.shape[-1]
    k = max(1, int(cfg.sparsity_ratio * T))

    # ── 5. PGD + TopK optimisation loop ──────────────────────────────────
    delta = torch.zeros_like(x)
    delta.requires_grad_(True)
    history: list[dict] = []

    pbar = tqdm(
        range(cfg.num_iter),
        desc="X-Shift",
        unit="step",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for step in pbar:
        x_adv = torch.clamp(x + delta, -1.0, 1.0)

        # Second-order grad requires MATH backend (no flash/efficient attention backward)
        with sdpa_kernel([SDPBackend.MATH]):
            adv_logits, adv_sal = _compute_saliency(
                model, x_adv, pred_orig, create_graph=True
            )
        adv_sal_flat = _flatten_normalize(adv_sal)

        # L_xshift: cosine distance to target (minimise → sal_adv aligns to target)
        cos_to_target = (adv_sal_flat * target_sal_flat).sum(dim=1)
        l_xshift = (1.0 - cos_to_target).mean()

        # L_pred: KL divergence — preserve original prediction distribution
        if adv_logits.shape[1] == 1:
            logit = adv_logits.squeeze(1)
            adv_log_probs = torch.stack(
                [F.logsigmoid(-logit), F.logsigmoid(logit)], dim=1
            )
        else:
            adv_log_probs = F.log_softmax(adv_logits, dim=-1)
        l_pred = F.kl_div(adv_log_probs, orig_probs, reduction="batchmean")

        total_loss = l_xshift + cfg.lambda_pred * l_pred

        if delta.grad is not None:
            delta.grad.zero_()
        model.zero_grad(set_to_none=True)
        total_loss.backward()

        if cfg.log_every and step % cfg.log_every == 0:
            with torch.no_grad():
                cos_orig_adv_now = (sal_orig_flat * adv_sal_flat.detach()).sum(dim=1)
                l0_frac = (delta.detach().abs() > 1e-9).float().mean().item()
            entry = {
                "step":          step,
                "loss":          total_loss.item(),
                "loss_xshift":   l_xshift.item(),
                "loss_pred":     l_pred.item(),
                "cos_to_target": cos_to_target.detach().mean().item(),
                "cos_orig_adv":  cos_orig_adv_now.mean().item(),
                "l0_frac":       l0_frac,
            }
            history.append(entry)
            pbar.set_postfix(
                loss=f"{entry['loss']:.4f}",
                xshift=f"{entry['loss_xshift']:.4f}",
                cos_t=f"{entry['cos_to_target']:.4f}",
                pred=f"{entry['loss_pred']:.4f}",
            )

        # Signed gradient descent + L∞ projection + L0 sparsity projection
        with torch.no_grad():
            step_dir = delta.grad.sign() if delta.grad is not None else torch.zeros_like(delta)
            delta_new = torch.clamp(delta - cfg.alpha * step_dir, -cfg.eps, cfg.eps)
            delta_new = _topk_sparsity(delta_new, k)

        del x_adv, adv_logits, adv_sal, adv_sal_flat, cos_to_target
        del l_xshift, l_pred, total_loss

        delta = delta_new.detach().requires_grad_(True)

    pbar.close()
    torch.cuda.empty_cache()

    # ── 6. Final evaluation ───────────────────────────────────────────────
    x_adv_final = torch.clamp(x + delta, -1.0, 1.0).detach()

    x_final_ref = x_adv_final.detach().requires_grad_(True)
    with torch.enable_grad():
        logits_final, sal_adv_final = _compute_saliency(
            model, x_final_ref, pred_orig, create_graph=False
        )
    sal_adv_final = sal_adv_final.detach()

    if logits_final.shape[1] == 1:
        pred_final = (logits_final.squeeze(1) >= 0).long()
    else:
        pred_final = logits_final.argmax(dim=-1)
    prediction_preserved = pred_final.detach() == pred_orig

    sal_adv_flat_final = _flatten_normalize(sal_adv_final)
    cos_to_target_final = (sal_adv_flat_final * target_sal_flat).sum(dim=1)
    cos_orig_adv_final  = (sal_orig_flat * sal_adv_flat_final).sum(dim=1)

    return XShiftResult(
        x_adv=x_adv_final,
        delta=delta.detach(),
        sal_original=sal_orig,
        sal_adv=sal_adv_final,
        target_sal=target_sal,
        cos_to_target=cos_to_target_final.detach(),
        cos_orig_adv=cos_orig_adv_final.detach(),
        prediction_preserved=prediction_preserved,
        history=history,
    )
