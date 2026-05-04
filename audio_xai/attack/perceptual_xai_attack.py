"""Perceptual XAI attack: change the explanation, not the prediction, inaudibly.

This is the core research contribution of the project. Given a trained
classifier and an input waveform, we find a perturbation δ such that:

    1. argmax f(x + δ) == argmax f(x)        (prediction preserved)
    2. GradCAM(x + δ) is far from GradCAM(x)  (explanation flipped)
    3. δ is below the psychoacoustic masking threshold of x (inaudible)

Loss landscape:

    L(δ) = L_explain(δ)       # cosine similarity, want it small
         + λ_aud * L_audibility(δ)
         + λ_pred * L_pred_preserve(δ)

Attack does NOT use PGD/FGSM — those work for first-order classification
attacks but are unstable when the loss includes second-order gradients (we
backprop through Grad-CAM, which itself contains a gradient). Adam is the
right tool here.

References
----------
- Ghorbani et al. 2019, "Interpretation of Neural Networks is Fragile"
- Heo et al. 2019, "Fooling Neural Network Interpretations via Adversarial Model Manipulation"
- Qin et al. 2019, "Imperceptible, Robust, and Targeted Adversarial Examples for ASR"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from audio_xai.metrics.psychoacoustic import (
    masking_threshold,
    perturbation_audibility_loss,
)
from audio_xai.models.base import AudioClassifier
from audio_xai.xai.gradcam import GradCAMBase, make_gradcam


@dataclass
class AttackConfig:
    n_steps: int = 200
    lr: float = 1e-3
    lambda_audibility: float = 1.0
    lambda_pred: float = 100.0
    pred_margin: float = 1.0
    linf_bound: float = 0.01  # hard L∞ bound on δ in waveform amplitude
    sample_rate: int = 16_000

    # Logging cadence — ``None`` disables.
    log_every: int | None = 20


@dataclass
class AttackResult:
    x_adv: torch.Tensor  # perturbed waveform [B, T]
    delta: torch.Tensor  # perturbation [B, T]
    cam_original: torch.Tensor  # heatmap on x
    cam_adv: torch.Tensor  # heatmap on x_adv
    cosine_similarity: torch.Tensor  # final cosine sim (closer to 0/-1 = better attack)
    prediction_preserved: torch.Tensor  # bool tensor [B]
    history: list[dict]  # per-step loss values for analysis


def _flatten_normalize(cam: torch.Tensor) -> torch.Tensor:
    """Flatten heatmap and L2-normalize; for cosine similarity."""
    flat = cam.flatten(start_dim=1)
    return F.normalize(flat, p=2, dim=1, eps=1e-8)


def perceptual_xai_attack(
    model: AudioClassifier,
    x: torch.Tensor,
    cfg: AttackConfig | None = None,
    gradcam: GradCAMBase | None = None,
) -> AttackResult:
    """Run the perceptual XAI attack on a batch of waveforms.

    Parameters
    ----------
    model : trained AudioClassifier in eval mode (we don't put it in eval
        mode here — caller's responsibility, since dropout/BN choices affect
        what "the model" means for the attack).
    x : [B, T] waveform tensor on the same device as the model.
    cfg : attack hyperparameters.
    gradcam : optional pre-built Grad-CAM. If None, picks the variant matching
        the model. Pass your own if you want a non-default target layer.
    """
    cfg = cfg or AttackConfig()
    if gradcam is None:
        gradcam = make_gradcam(model)

    # ---- Setup: original prediction, original explanation, masking floor ----
    with torch.no_grad():
        logits_orig = model(x)
        pred_orig = logits_orig.argmax(dim=-1)
        threshold_db = masking_threshold(x, sample_rate=cfg.sample_rate)

    # Original CAM is computed once and detached — it's our reference target.
    cam_original = gradcam(x, target_class=pred_orig, create_graph=False).detach()
    cam_orig_flat = _flatten_normalize(cam_original)

    # ---- Optimization variable ----
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.lr)

    history: list[dict] = []

    for step in range(cfg.n_steps):
        x_adv = x + delta

        # 1. Explanation loss: minimize cosine similarity to the original CAM.
        cam_adv = gradcam(x_adv, target_class=pred_orig, create_graph=True)
        cam_adv_flat = _flatten_normalize(cam_adv)
        cos_sim = (cam_orig_flat * cam_adv_flat).sum(dim=1)
        loss_explain = cos_sim.mean()

        # 2. Audibility loss: only counts perturbation above masking threshold.
        loss_aud = perturbation_audibility_loss(delta, threshold_db, sample_rate=cfg.sample_rate)

        # 3. Prediction-preserving hinge: penalize only when the wrong class
        #    is within ``pred_margin`` of the right class.
        logits_adv = model(x_adv)
        correct = logits_adv.gather(1, pred_orig.view(-1, 1)).squeeze(1)
        # For binary, "other" is just 1 - pred; generalize with masking for n>2.
        mask = F.one_hot(pred_orig, num_classes=logits_adv.shape[1]).bool()
        other = logits_adv.masked_fill(mask, float("-inf")).max(dim=-1).values
        loss_pred = F.relu(other - correct + cfg.pred_margin).mean()

        loss = loss_explain + cfg.lambda_audibility * loss_aud + cfg.lambda_pred * loss_pred

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Hard L∞ projection — sanity bound on top of the perceptual constraint.
        with torch.no_grad():
            delta.clamp_(-cfg.linf_bound, cfg.linf_bound)

        if cfg.log_every and step % cfg.log_every == 0:
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "loss_explain": loss_explain.item(),
                    "loss_audibility": loss_aud.item(),
                    "loss_pred": loss_pred.item(),
                    "cos_sim": cos_sim.detach().mean().item(),
                }
            )

    # ---- Final evaluation ----
    with torch.no_grad():
        x_adv_final = (x + delta).detach()
        logits_final = model(x_adv_final)
        pred_final = logits_final.argmax(dim=-1)
        prediction_preserved = pred_final == pred_orig

    # Grad-CAM requires gradients to compute even at eval time (it differentiates
    # the model output w.r.t. activations). Only the resulting heatmap is detached.
    cam_adv_final = gradcam(x_adv_final, target_class=pred_orig, create_graph=False).detach()
    with torch.no_grad():
        cos_sim_final = (_flatten_normalize(cam_original) * _flatten_normalize(cam_adv_final)).sum(dim=1)

    gradcam.remove_hooks()

    return AttackResult(
        x_adv=x_adv_final,
        delta=delta.detach(),
        cam_original=cam_original,
        cam_adv=cam_adv_final,
        cosine_similarity=cos_sim_final,
        prediction_preserved=prediction_preserved,
        history=history,
    )


# ---- Evaluation helpers (use these for your results tables, not optimization) ----


def topk_overlap(cam_a: torch.Tensor, cam_b: torch.Tensor, k_frac: float = 0.1) -> torch.Tensor:
    """Jaccard overlap of top-k% pixels between two heatmaps.

    Range [0, 1]. Lower = more disagreement = more successful attack.
    """
    B = cam_a.shape[0]
    flat_a = cam_a.flatten(start_dim=1)
    flat_b = cam_b.flatten(start_dim=1)
    n_pixels = flat_a.shape[1]
    k = max(1, int(k_frac * n_pixels))

    _, idx_a = flat_a.topk(k, dim=1)
    _, idx_b = flat_b.topk(k, dim=1)

    overlaps = []
    for i in range(B):
        sa = set(idx_a[i].tolist())
        sb = set(idx_b[i].tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        overlaps.append(inter / union if union > 0 else 0.0)
    return torch.tensor(overlaps)


def heatmap_ssim(cam_a: torch.Tensor, cam_b: torch.Tensor) -> torch.Tensor:
    """Simple per-sample SSIM between two heatmaps. Lower = more change.

    Minimal implementation — for publication-grade SSIM use ``torchmetrics``
    or ``pytorch-msssim``.
    """
    a = cam_a.flatten(start_dim=1)
    b = cam_b.flatten(start_dim=1)
    mu_a, mu_b = a.mean(dim=1, keepdim=True), b.mean(dim=1, keepdim=True)
    var_a = ((a - mu_a) ** 2).mean(dim=1)
    var_b = ((b - mu_b) ** 2).mean(dim=1)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=1)
    c1, c2 = 1e-4, 9e-4
    num = (2 * mu_a.squeeze() * mu_b.squeeze() + c1) * (2 * cov + c2)
    den = (mu_a.squeeze() ** 2 + mu_b.squeeze() ** 2 + c1) * (var_a + var_b + c2)
    return num / den
