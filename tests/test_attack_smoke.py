"""Smoke test for the perceptual XAI attack.

Runs the attack on a random batch with an *untrained* VGGish, just to verify
that:
    - Grad-CAM hooks register and fire
    - Second-order gradients flow through to ``delta``
    - All three losses (explain, audibility, prediction) compute without error
    - The attack loop completes and returns an AttackResult

Does NOT verify that the attack is *successful* — for that you need a trained
classifier and real audio.

Usage: python tests/test_attack_smoke.py
"""

from __future__ import annotations

import torch

from audio_xai.attacks.perceptual_xai_attack import (
    AttackConfig,
    perceptual_xai_attack,
    topk_overlap,
)
from audio_xai.models.vggish_binary import VGGishBinary


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = VGGishBinary().to(device).eval()

    # Random "audio": batch of 2 clips, 1 second at 16 kHz.
    x = 0.1 * torch.randn(2, 16_000, device=device)

    cfg = AttackConfig(n_steps=10, log_every=2, lambda_audibility=0.1)
    print(f"Running attack: {cfg.n_steps} steps...")

    result = perceptual_xai_attack(model, x, cfg)

    print(f"\nFinal cosine similarity: {result.cosine_similarity.tolist()}")
    print(f"Predictions preserved:   {result.prediction_preserved.tolist()}")
    print(f"Δ L∞ norm:              {result.delta.abs().max().item():.4f}")
    print(f"Δ L2 norm:              {result.delta.norm(dim=-1).tolist()}")
    overlap = topk_overlap(result.cam_original, result.cam_adv, k_frac=0.1)
    print(f"Top-10% overlap:        {overlap.tolist()}")

    print("\nLoss history:")
    for entry in result.history:
        print(
            f"  step {entry['step']:3d}: "
            f"L={entry['loss']:.4f} "
            f"explain={entry['loss_explain']:.4f} "
            f"aud={entry['loss_audibility']:.4f} "
            f"pred={entry['loss_pred']:.4f}"
        )

    assert result.x_adv.shape == x.shape
    assert result.delta.shape == x.shape
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
