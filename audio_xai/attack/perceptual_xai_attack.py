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

References:
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
    """Hyperparameters controlling the perceptual XAI attack optimization loop.

    Attributes:
        n_steps: Number of Adam optimizer iterations.
        lr: Adam learning rate for the perturbation delta.
        lambda_audibility: Weight for the psychoacoustic audibility loss term.
        lambda_pred: Weight for the prediction-preservation hinge loss term.
        pred_margin: Hinge margin; penalty activates when the non-target logit is within this value of the target logit.
        linf_bound: Hard L-infinity clamp applied to delta after each step (waveform amplitude units).
        sample_rate: Audio sample rate in Hz; used by the psychoacoustic masking model.
        log_every: Log loss values every this many steps; set to ``None`` to disable logging.
    """

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
    """Output container returned by :func:`perceptual_xai_attack`.

    Attributes:
        x_adv: Adversarial waveform with shape [B, T] (original input plus optimized delta).
        delta: Final perturbation tensor with shape [B, T] added to the original input.
        cam_original: Grad-CAM heatmap computed on the original, unperturbed inputs.
        cam_adv: Grad-CAM heatmap computed on the adversarial inputs after optimization.
        cosine_similarity: Per-sample cosine similarity between original and adversarial CAMs; values closer to 0 or -1 indicate a more successful attack.
        prediction_preserved: Boolean tensor of shape [B] indicating whether each sample's predicted class was unchanged by the perturbation.
        history: List of per-step dictionaries with logged loss values (populated every ``cfg.log_every`` steps).
    """

    x_adv: torch.Tensor  # perturbed waveform [B, T]
    delta: torch.Tensor  # perturbation [B, T]
    cam_original: torch.Tensor  # heatmap on x
    cam_adv: torch.Tensor  # heatmap on x_adv
    cosine_similarity: torch.Tensor  # final cosine sim (closer to 0/-1 = better attack)
    prediction_preserved: torch.Tensor  # bool tensor [B]
    history: list[dict]  # per-step loss values for analysis


def _flatten_normalize(cam: torch.Tensor) -> torch.Tensor:
    """Flatten a batch of heatmaps and L2-normalize each sample along the feature dimension.

    Parameters:
        cam (torch.Tensor): Batch of heatmaps with shape [B, ...] (B = batch size).

    Returns:
        torch.Tensor: Tensor of shape [B, N] where each row is the L2-normalized flattened heatmap (`eps=1e-8` used for numerical stability).
    """
    flat = cam.flatten(start_dim=1)
    return F.normalize(flat, p=2, dim=1, eps=1e-8)


def perceptual_xai_attack(
    model: AudioClassifier,
    x: torch.Tensor,
    cfg: AttackConfig | None = None,
    gradcam: GradCAMBase | None = None,
) -> AttackResult:
    """Run the perceptual XAI adversarial attack on a batch of waveforms.

    Parameters:
        model (AudioClassifier): The trained audio classifier (caller should set model to the desired mode; evaluation mode is recommended because dropout/BN affect behaviour).
        x (torch.Tensor): Input waveforms of shape [B, T] on the same device as the model.
        cfg (AttackConfig | None): Attack hyperparameters; defaults to AttackConfig() when None.
        gradcam (GradCAMBase | None): Optional pre-built Grad-CAM instance; if None a suitable Grad-CAM for the model is created.

    Returns:
        AttackResult: Container with fields:
            - x_adv: adversarial waveform tensor [B, T].
            - delta: final perturbation tensor added to x.
            - cam_original: Grad-CAM heatmap for the original inputs.
            - cam_adv: Grad-CAM heatmap for the adversarial inputs.
            - cosine_similarity: per-sample cosine similarity between original and adversarial CAMs.
            - prediction_preserved: boolean tensor indicating whether each sample's predicted class was preserved.
            - history: list of per-step logged loss dictionaries.
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
    """Compute the per-sample Jaccard (intersection-over-union) overlap between the top-k fraction of pixels in two heatmaps.

    k is determined as max(1, int(k_frac * N_pixels)); overlap values lie in [0.0, 1.0].

    Returns:
        overlaps (torch.Tensor): 1-D tensor of length B containing the Jaccard overlap for each sample; each value is intersection_size / union_size (0.0 if union is 0).
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
    """Compute a lightweight per-sample SSIM-like similarity between two heatmaps.

    Flattens each heatmap per batch item and computes per-sample means, variances,
    and covariance to form a simplified SSIM score. This is a minimal approximation
    for comparing explanation heatmaps; it is not a publication-grade SSIM.

    Parameters:
        cam_a (torch.Tensor): First batch of heatmaps with shape [B, ...].
        cam_b (torch.Tensor): Second batch of heatmaps with shape [B, ...].

    Returns:
        ssim (torch.Tensor): Per-sample similarity scores with shape [B]; lower
        values indicate greater change between corresponding heatmaps.
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
