"""PGD-based XAI attack using input-gradient saliency maps.

Unlike the Adam-based perceptual_xai_attack (which uses Grad-CAM hooks),
this attack uses signed projected gradient descent with input-gradient
saliency as the explanation map.

Loss:
    L(δ) = cos_sim(sal(x), sal(x+δ))       — minimise = saliency diverges
          + λ_aud  · ‖δ‖²₂ / T             — L2 audibility penalty
          + λ_pred · KL(f(x+δ) ‖ f(x))     — preserve prediction distribution

PGD update (signed-gradient, L∞-projected):
    δ ← clamp(δ − α · sign(∇_δ L),  −ε, +ε)

Saliency: |d score_i / d x_i| computed via torch.autograd.grad with
create_graph=True so the gradient is itself differentiable w.r.t. δ
(second-order differentiation through the model is required).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from audio_xai.models.base import AudioClassifier


@dataclass
class PGDAttackConfig:
    eps: float = 0.01           # L∞ radius of the perturbation ball
    alpha: float = 0.001        # signed-gradient step size per iteration
    num_iter: int = 100         # number of PGD iterations
    lambda_aud: float = 0.5     # L2 audibility regulariser weight
    lambda_pred: float = 100.0  # KL prediction-preservation weight
    sample_rate: int = 16_000
    log_every: int | None = 20


@dataclass
class PGDAttackResult:
    x_adv: torch.Tensor               # perturbed waveform [B, T]
    delta: torch.Tensor               # perturbation [B, T]
    sal_original: torch.Tensor        # |∂ score / ∂ x| on clean input [B, T]
    sal_adv: torch.Tensor             # |∂ score / ∂ x| on adversarial input [B, T]
    cosine_similarity: torch.Tensor   # per-sample cosine similarity [B]
    prediction_preserved: torch.Tensor  # bool tensor [B]
    history: list[dict]


def _class_score(logits: torch.Tensor, pred_class: torch.Tensor) -> torch.Tensor:
    """Predicted-class score per sample; signed for binary single-logit outputs."""
    if logits.shape[1] == 1:
        scores = logits.squeeze(1)
        return torch.where(pred_class == 1, scores, -scores)
    return logits.gather(1, pred_class.view(-1, 1)).squeeze(1)


def _flatten_normalize(t: torch.Tensor) -> torch.Tensor:
    flat = t.flatten(start_dim=1)
    return F.normalize(flat, p=2, dim=1, eps=1e-8)


def _compute_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    pred_class: torch.Tensor,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass + input-gradient saliency.

    Returns (logits [B, C], |d score / d x| [B, T]).

    When create_graph=True the saliency tensor is differentiable w.r.t. any
    leaf tensors that x depends on (i.e. delta in the PGD loop).  PyTorch
    sets retain_graph=True implicitly when create_graph=True, so logits is
    still valid for further use (e.g. KL loss) after this call.
    """
    logits = model(x)
    score_sum = _class_score(logits, pred_class).sum()
    (grad,) = torch.autograd.grad(score_sum, x, create_graph=create_graph)
    return logits, grad.abs()


def pgd_xai_attack(
    model: AudioClassifier,
    x: torch.Tensor,
    cfg: PGDAttackConfig | None = None,
) -> PGDAttackResult:
    """Run the PGD XAI attack on a batch of waveforms.

    Args:
        model: Trained AudioClassifier in eval mode.
        x: Input waveforms [B, T] on the model's device.
        cfg: PGD hyperparameters; defaults to PGDAttackConfig().

    Returns:
        PGDAttackResult with adversarial waveforms, saliency maps, and metrics.
    """
    cfg = cfg or PGDAttackConfig()

    # ── 1. Baseline predictions and reference probability distribution ─────
    with torch.no_grad():
        logits_orig = model(x)
        if logits_orig.shape[1] == 1:
            pred_orig = (logits_orig.squeeze(1) >= 0).long()
            p = torch.sigmoid(logits_orig)
            orig_probs = torch.cat([1 - p, p], dim=1)           # [B, 2]
        else:
            pred_orig = logits_orig.argmax(dim=-1)
            orig_probs = F.softmax(logits_orig, dim=-1)          # [B, C]

    # ── 2. Reference saliency on clean input (detached — no need for grad) ─
    x_clean = x.detach().requires_grad_(True)
    with torch.enable_grad():
        _, sal_orig = _compute_saliency(model, x_clean, pred_orig, create_graph=False)
    sal_orig = sal_orig.detach()
    sal_orig_flat = _flatten_normalize(sal_orig)
    del x_clean

    # ── 3. PGD optimisation loop ───────────────────────────────────────────
    # Initialise delta randomly inside the L∞ ball (standard PGD start)
    delta = torch.zeros_like(x).uniform_(-cfg.eps, cfg.eps)
    delta.requires_grad_(True)
    history: list[dict] = []

    pbar = tqdm(
        range(cfg.num_iter),
        desc="PGD",
        unit="step",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for step in pbar:
        x_adv = torch.clamp(x + delta, -1.0, 1.0)

        # create_graph=True: adv_sal and adv_logits are differentiable w.r.t. delta.
        # MATH backend required: flash/efficient attention don't implement the
        # second-order backward that goes through autograd.grad(..., create_graph=True).
        with sdpa_kernel([SDPBackend.MATH]):
            adv_logits, adv_sal = _compute_saliency(
                model, x_adv, pred_orig, create_graph=True
            )
        adv_sal_flat = _flatten_normalize(adv_sal)

        # L_explain: cosine similarity (minimise → explanation diverges)
        cos_sim = (sal_orig_flat * adv_sal_flat).sum(dim=1)
        l_explain = cos_sim.mean()

        # L_audibility: mean L2 energy of the perturbation
        l_aud = delta.pow(2).mean()

        # L_pred: KL divergence — preserve prediction distribution
        if adv_logits.shape[1] == 1:
            logit = adv_logits.squeeze(1)
            adv_log_probs = torch.stack(
                [F.logsigmoid(-logit), F.logsigmoid(logit)], dim=1
            )
        else:
            adv_log_probs = F.log_softmax(adv_logits, dim=-1)
        l_pred = F.kl_div(adv_log_probs, orig_probs, reduction="batchmean")

        total_loss = l_explain + cfg.lambda_aud * l_aud + cfg.lambda_pred * l_pred

        if delta.grad is not None:
            delta.grad.zero_()
        model.zero_grad(set_to_none=True)
        total_loss.backward()

        if cfg.log_every and step % cfg.log_every == 0:
            entry = {
                "step":            step,
                "loss":            total_loss.item(),
                "loss_explain":    l_explain.item(),
                "loss_audibility": l_aud.item(),
                "loss_pred":       l_pred.item(),
                "cos_sim":         cos_sim.detach().mean().item(),
            }
            history.append(entry)
            pbar.set_postfix(
                loss=f"{entry['loss']:.4f}",
                explain=f"{entry['loss_explain']:.4f}",
                cos=f"{entry['cos_sim']:.4f}",
                pred=f"{entry['loss_pred']:.4f}",
            )

        # Signed gradient descent + L∞ ball projection
        with torch.no_grad():
            step_dir = delta.grad.sign() if delta.grad is not None else torch.zeros_like(delta)
            delta_new = torch.clamp(delta - cfg.alpha * step_dir, -cfg.eps, cfg.eps)

        del x_adv, adv_logits, adv_sal, adv_sal_flat, cos_sim
        del l_explain, l_aud, l_pred, total_loss

        delta = delta_new.detach().requires_grad_(True)

    pbar.close()
    torch.cuda.empty_cache()

    # ── 4. Final evaluation ────────────────────────────────────────────────
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

    cos_sim_final = (
        _flatten_normalize(sal_orig) * _flatten_normalize(sal_adv_final)
    ).sum(dim=1)

    return PGDAttackResult(
        x_adv=x_adv_final,
        delta=delta.detach(),
        sal_original=sal_orig,
        sal_adv=sal_adv_final,
        cosine_similarity=cos_sim_final.detach(),
        prediction_preserved=prediction_preserved,
        history=history,
    )