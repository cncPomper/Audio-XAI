"""Layer-Wise Relevance Propagation (LRP) for audio classification models.

Two implementations behind a common interface:

CNNLRP — epsilon-LRP propagated layer-by-layer through VGGish.
  Rule:  R_j = Σ_k (a_j·w_jk) / (z_k + ε·sign(z_k)) · R_k
  Propagation continues through all convolutional blocks back to the input
  log-mel spectrogram, giving a [B, 96, 64] heatmap at full patch resolution
  (one relevance value per mel bin × time frame).

TransformerLRP — AttnLRP / Gradient×Input for AST and SpecTra.
  Implements the first-order Taylor expansion of LRP, which equals
  Gradient×Input (input · ∂score/∂input) summed over the embedding dimension.
  This is equivalent to epsilon-LRP in the linear approximation and is the
  recommended approach for attention-based models (Achtibat et al., ICML 2024).

References:
  Bach et al. (2015) — PLoS ONE 10(7):e0130140.
  Achtibat et al. (2024) — AttnLRP, ICML 2024. arXiv:2402.05602.
  Draelos & Carin (2021) — MDPI Information 2021, 15(12), 751.
  https://github.com/rachtibat/LRP-eXplains-Transformers
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_xai.models.base import AudioClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Numerical helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stabilize(z: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard epsilon-LRP stabilizer: preserves sign, ensures |denom| ≥ eps.

    Maps z → z + eps·sign(z), where sign(0) = +1 by convention.
    This guarantees the denominator never crosses zero, while keeping
    the sign of the relevance redistribution consistent with z.
    """
    sign = z.sign()
    sign[sign == 0] = 1.0
    return z + eps * sign


def _lrp_linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    R: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Epsilon-LRP through a linear layer  y = aW^T + b.

    Args:
        a:      Pre-layer activations     [*, in_features]
        weight: Layer weight matrix       [out_features, in_features]
        bias:   Layer bias                [out_features] or None
        R:      Incoming relevance        [*, out_features]
        eps:    Stabilizer constant

    Returns:
        Redistributed relevance           [*, in_features]
    """
    z = F.linear(a, weight, bias)
    s = R / _stabilize(z, eps)      # [*, out]
    c = s @ weight                  # [*, in]  — gradient of z w.r.t. a
    return a * c


def _lrp_conv(
    a: torch.Tensor,
    layer: nn.Conv2d,
    R: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Epsilon-LRP through a Conv2d layer."""
    z = F.conv2d(a, layer.weight, layer.bias, layer.stride, layer.padding)
    s = R / _stabilize(z, eps)
    c = F.conv_transpose2d(
        s, layer.weight, stride=layer.stride, padding=layer.padding
    )
    return a * c


def _lrp_maxpool(
    a: torch.Tensor,
    R: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Route LRP relevance to the maximum element in each pool window.

    Uses the gradient of MaxPool2d, which is an all-or-nothing selector:
    the gradient passes only to the selected maximum.
    """
    a_d = a.detach().requires_grad_(True)
    with torch.enable_grad():
        z = F.max_pool2d(a_d, kernel_size=2, stride=2)
        (R.detach() / _stabilize(z, eps)).sum().backward()
    return (a_d.grad * a.detach()).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class LRPBase(ABC):
    """Common interface: call with a waveform, get a relevance heatmap [B, H, W]."""

    def __init__(self, model: AudioClassifier, eps: float = 1e-6):
        self.model = model
        self.eps = eps

    @abstractmethod
    def __call__(
        self,
        waveform: torch.Tensor,
        target_class: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Return a non-negative relevance heatmap [B, H, W]."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# VGGish — epsilon-LRP through Conv + FC layers
# ──────────────────────────────────────────────────────────────────────────────

class CNNLRP(LRPBase):
    """Epsilon-LRP for VGGishBinary.

    Propagates relevance from the output logit backward through the linear
    embedding head and all convolutional blocks to the input log-mel
    spectrogram, giving a [B, 96, 64] heatmap at full patch resolution.
    """

    def __call__(
        self,
        waveform: torch.Tensor,
        target_class: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        model = self.model
        eps = self.eps

        # ── Features ─────────────────────────────────────────────────────────
        with torch.no_grad():
            patches = model.waveform_to_features(waveform)  # [B, P, 1, 96, 64]
        B, P = patches.shape[:2]
        x = patches.reshape(B * P, 1, 96, 64)

        # Forward through features, store activations before each Conv2d.
        conv_acts: list[torch.Tensor] = []   # [a_before_conv] for each Conv2d
        pool_acts: list[torch.Tensor] = []   # [a_before_pool] for each MaxPool2d

        for layer in model.backbone.features:
            if isinstance(layer, nn.Conv2d):
                conv_acts.append(x.detach())
                x = F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding)
            elif isinstance(layer, nn.ReLU):
                x = x.clamp(min=0)           # in-place-safe ReLU
            elif isinstance(layer, nn.MaxPool2d):
                pool_acts.append(x.detach())
                x = F.max_pool2d(x, 2, 2)

        # x: [B*P, 512, 6, 4] — output of last MaxPool

        # ── Embedding head — forward ──────────────────────────────────────────
        emb = model.backbone.embedding
        lin1: nn.Linear = emb[1]  # Linear(12288, 4096)
        lin2: nn.Linear = emb[3]  # Linear(4096, 4096)
        lin3: nn.Linear = emb[5]  # Linear(4096, 128)

        a_flat = x.reshape(B * P, -1)                          # [B*P, 12288]
        a_fc1  = F.relu(F.linear(a_flat, lin1.weight, lin1.bias))  # [B*P, 4096]
        a_fc2  = F.relu(F.linear(a_fc1,  lin2.weight, lin2.bias))  # [B*P, 4096]
        a_fc3  = F.linear(a_fc2, lin3.weight, lin3.bias)            # [B*P, 128]

        logits_patch = F.linear(a_fc3, model.head.weight, model.head.bias)  # [B*P, C]
        logits = logits_patch.reshape(B, P, -1).mean(dim=1)                 # [B, C]

        # ── Target class ──────────────────────────────────────────────────────
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        if isinstance(target_class, int):
            target_class = torch.full(
                (B,), target_class, dtype=torch.long, device=waveform.device
            )

        # ── Initial relevance: one-hot at target class → per-patch ───────────
        R = torch.zeros_like(logits)
        R.scatter_(1, target_class.view(-1, 1), 1.0)
        R = R.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1) / P  # [B*P, C]

        # ── LRP backward through head and embedding ───────────────────────────
        R = _lrp_linear(a_fc3,  model.head.weight, model.head.bias,  R, eps)
        R = _lrp_linear(a_fc2,  lin3.weight,       lin3.bias,         R, eps)
        R = _lrp_linear(a_fc1,  lin2.weight,       lin2.bias,         R, eps)
        R = _lrp_linear(a_flat, lin1.weight,        lin1.bias,         R, eps)

        # ── Unflatten: [B*P, 12288] → [B*P, 512, 6, 4] ──────────────────────
        R = R.reshape(B * P, 512, 6, 4)

        # ── LRP backward through conv + pool layers → input spectrogram ──────
        # Consume stored activations in reverse order, mirroring the forward pass.
        conv_acts_rev = list(reversed(conv_acts))
        pool_acts_rev = list(reversed(pool_acts))
        conv_idx = pool_idx = 0

        for layer in reversed([l for l in model.backbone.features
                                if isinstance(l, (nn.Conv2d, nn.MaxPool2d))]):
            if isinstance(layer, nn.MaxPool2d):
                R = _lrp_maxpool(pool_acts_rev[pool_idx], R, eps)
                pool_idx += 1
            else:
                R = _lrp_conv(conv_acts_rev[conv_idx], layer, R, eps)
                conv_idx += 1

        # R: [B*P, 1, 96, 64] — relevance at the input log-mel spectrogram
        # VGGish patch layout: dim-2 = time frames (96), dim-3 = mel bands (64).
        # Transpose to [freq, time] = [64, 96] so the heatmap matches spectrogram
        # orientation (freq on height axis, time on width axis), consistent with
        # how AST returns [freq_patches, time_patches].
        heatmap = R.squeeze(1).clamp(min=0)               # [B*P, 96, 64]
        heatmap = heatmap.transpose(-2, -1)                # [B*P, 64, 96]

        # Average over patches → [B, 64, 96]
        return heatmap.reshape(B, P, 64, 96).mean(dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer (AST / SpecTra) — AttnLRP via Gradient × Input
# ──────────────────────────────────────────────────────────────────────────────

class TransformerLRP(LRPBase):
    """AttnLRP for ASTBinary and Wav2Vec2Binary.

    Computes relevance as  R = emb · (∂score/∂emb),  summed over the hidden
    dimension.  This is the first-order Taylor decomposition of LRP, and is
    the method recommended by Achtibat et al. for attention-based models
    because it correctly propagates through the softmax attention.

    For AST  : embeddings are [B, N_tokens, D]; special tokens are stripped
               and the patch grid is reshaped to [B, freq_patches, time_patches].
    For SpecTra (Wav2Vec2): the projected CNN features are [B, T', D]; the
               output is [B, 1, T'] — a 1-D time relevance map.
    """

    def __init__(
        self,
        model: AudioClassifier,
        eps: float = 1e-6,
        num_special_tokens: int = 2,   # AST: CLS + distillation tokens
        freq_patches: int = 12,
        time_patches: int = 101,
    ):
        super().__init__(model, eps)
        self.num_special = num_special_tokens
        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self._embeddings: torch.Tensor | None = None
        self._handles: list = []
        self._register_hook()

    def _embedding_layer(self) -> nn.Module:
        cls = type(self.model).__name__
        if cls == "ASTBinary":
            return self.model.backbone.audio_spectrogram_transformer.embeddings
        if cls == "Wav2Vec2Binary":
            return self.model.backbone.feature_projection
        raise ValueError(f"TransformerLRP: unsupported model type {cls!r}")

    def _register_hook(self) -> None:
        def fwd_hook(_m, _inp, out):
            # Wav2Vec2FeatureProjection returns (hidden, normed_hidden) tuple.
            self._embeddings = out[0] if isinstance(out, tuple) else out

        self._handles.append(
            self._embedding_layer().register_forward_hook(fwd_hook)
        )

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __exit__(self, *_):
        self.remove_hooks()

    # ------------------------------------------------------------------

    def __call__(
        self,
        waveform: torch.Tensor,
        target_class: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        self._embeddings = None

        logits = self.model(waveform)

        if target_class is None:
            target_class = logits.argmax(dim=-1)
        if isinstance(target_class, int):
            target_class = torch.full(
                (waveform.shape[0],), target_class,
                dtype=torch.long, device=waveform.device,
            )

        score = logits.gather(1, target_class.view(-1, 1)).sum()

        assert self._embeddings is not None, \
            "Forward hook did not fire — check _embedding_layer()"

        # ∂score/∂embeddings
        grads = torch.autograd.grad(
            score,
            self._embeddings,
            create_graph=False,
            retain_graph=False,
        )[0]  # [B, N, D]

        # AttnLRP: relevance = embedding × gradient, sum over hidden dim
        relevance = (self._embeddings.detach() * grads).sum(dim=-1)  # [B, N]

        return self._to_heatmap(relevance, waveform.device)

    def _to_heatmap(
        self, relevance: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Reshape token relevance [B, N] into a 2-D spatial heatmap [B, H, W]."""
        cls = type(self.model).__name__
        B = relevance.shape[0]

        if cls == "ASTBinary":
            # Drop CLS + distillation tokens, reshape to freq × time grid.
            patch_rel = relevance[:, self.num_special:]  # [B, N_patches]
            N = patch_rel.shape[1]
            F_p = self.freq_patches
            T_p = N // F_p
            patch_rel = patch_rel[:, : F_p * T_p].reshape(B, F_p, T_p)
            return patch_rel.clamp(min=0)

        if cls == "Wav2Vec2Binary":
            # Wav2Vec2 output is [B, T'] — return as [B, 1, T'] for 2-D compat.
            return relevance.clamp(min=0).unsqueeze(1)  # [B, 1, T']

        raise ValueError(f"_to_heatmap: unsupported model type {cls!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Sonics (SpecTTTra) — AttnLRP via Gradient × Input on STTokenizer output
# ──────────────────────────────────────────────────────────────────────────────

class SonicsLRP(TransformerLRP):
    """AttnLRP for SonicsWrapper (SpecTTTra encoder).

    Hooks the STTokenizer output [B, N_tokens, D] where
    N_tokens = temporal_tokens + spectral_tokens (no special tokens).
    Returns [B, 1, N_tokens] so lrp_chunked can concatenate along the time axis.
    """

    def __init__(self, model: AudioClassifier, eps: float = 1e-6):
        super().__init__(model, eps, num_special_tokens=0)

    def _embedding_layer(self) -> nn.Module:
        return self.model._m.encoder.st_tokenizer

    def _to_heatmap(self, relevance: torch.Tensor, device: torch.device) -> torch.Tensor:
        return relevance.clamp(min=0).unsqueeze(1)  # [B, 1, N_tokens]


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_lrp(model: AudioClassifier, eps: float = 1e-6) -> LRPBase:
    """Return the correct LRP variant for the given model."""
    cls = type(model).__name__
    if cls == "VGGishBinary":
        return CNNLRP(model, eps=eps)
    if cls == "ASTBinary":
        return TransformerLRP(model, eps=eps)
    if cls == "Wav2Vec2Binary":
        return TransformerLRP(model, eps=eps)
    if cls == "SonicsWrapper":
        return SonicsLRP(model, eps=eps)
    raise ValueError(
        f"No LRP variant registered for {cls}. "
        "Add it to make_lrp() or subclass LRPBase directly."
    )