"""Grad-CAM with second-order gradient support.

Why we don't use Captum here: ``LayerGradCam.attribute`` has a
``create_graph`` kwarg, but it doesn't propagate through all internal ops
reliably for transformer layers, and we need rock-solid second-order
gradients for the attack loop. Hand-rolled hooks are 60 lines and behave
predictably.

Two variants behind the same interface:
    - ``CNNGradCAM``: classic spatial CNN Grad-CAM (for VGGish).
    - ``TransformerGradCAM``: token-axis Grad-CAM reshaped back to (freq, time)
      grid (for AST).

Both return a heatmap shaped like the model's spectrogram input, so all
downstream code (distance metrics, visualization) is model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from audio_xai.models.base import AudioClassifier


class GradCAMBase(ABC):
    """Common Grad-CAM scaffolding.

    Subclasses define how to turn the (activations, gradients) pair from the
    target layer into a 2-D heatmap. The forward/backward bookkeeping is
    shared.
    """

    def __init__(self, model: AudioClassifier):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        layer = self.model.target_layer

        def fwd_hook(_module, _inputs, output):
            self._activations = output

        def bwd_hook(_module, _grad_in, grad_out):
            # grad_out is a tuple; first element is the gradient w.r.t. output.
            self._gradients = grad_out[0]

        self._handles.append(layer.register_forward_hook(fwd_hook))
        # full_backward_hook works with create_graph=True; the older
        # register_backward_hook does not.
        self._handles.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove_hooks()

    @abstractmethod
    def _build_heatmap(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor: ...

    def __call__(
        self,
        waveform: torch.Tensor,
        target_class: torch.Tensor | int | None = None,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Compute Grad-CAM for the given waveform.

        Parameters
        ----------
        waveform : [B, T] tensor.
        target_class : class index to explain. If None, uses argmax.
        create_graph : if True, the returned heatmap is differentiable w.r.t.
            ``waveform``. Required when the heatmap is used as part of a loss
            you backprop through (the attack loop).

        Returns
        -------
        Tensor [B, H, W] — heatmap per sample.
        """
        self._activations = None
        self._gradients = None

        logits = self.model(waveform)
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class] * waveform.shape[0], device=waveform.device)

        # Score = logit of target class, summed over batch.
        score = logits.gather(1, target_class.view(-1, 1)).sum()

        # The model's gradients w.r.t. its target layer's activations.
        # retain_graph=True so the caller can still backprop the heatmap loss.
        assert self._activations is not None, "Forward hook did not fire — check target_layer"
        grads = torch.autograd.grad(
            score,
            self._activations,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        return self._build_heatmap(self._activations, grads)


class CNNGradCAM(GradCAMBase):
    """Classic Grad-CAM for 4-D conv activations [B, C, H, W]."""

    def _build_heatmap(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        # Channel-wise weights = global-average-pooled gradients.
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1)  # [B, H, W]
        return F.relu(cam)


class TransformerGradCAM(GradCAMBase):
    """Grad-CAM for transformer activations [B, N_tokens, D].

    AST tokenizes the spectrogram into (freq_patches × time_patches) patches,
    plus a CLS token (and on AudioSet checkpoints, a distillation token too).
    We drop the special tokens, reshape to a 2-D grid, and treat the embedding
    dim as the "channel" dim for Grad-CAM weighting.
    """

    def __init__(
        self,
        model: AudioClassifier,
        num_special_tokens: int = 2,
        freq_patches: int = 12,
        time_patches: int = 101,
    ):
        super().__init__(model)
        self.num_special = num_special_tokens
        self.freq_patches = freq_patches
        self.time_patches = time_patches

    def _build_heatmap(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        # Drop CLS / distillation tokens.
        a = activations[:, self.num_special :, :]
        g = gradients[:, self.num_special :, :]

        B, N, D = a.shape
        expected = self.freq_patches * self.time_patches
        if N != expected:
            # Fall back to inferring a square-ish grid if the user's AST
            # has a different patch count (e.g. different clip length).
            # The attack still works; viz might be slightly off.
            t = N // self.freq_patches
            a = a[:, : self.freq_patches * t, :]
            g = g[:, : self.freq_patches * t, :]
            self.time_patches = t

        # Reshape: [B, freq_patches, time_patches, D]
        a = a.reshape(B, self.freq_patches, self.time_patches, D)
        g = g.reshape(B, self.freq_patches, self.time_patches, D)

        # Weights per embedding dim, averaged over the grid.
        weights = g.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, D]
        cam = (weights * a).sum(dim=-1)  # [B, F, T]
        return F.relu(cam)


def make_gradcam(model: AudioClassifier) -> GradCAMBase:
    """Return the right Grad-CAM variant for the given model."""
    cls_name = type(model).__name__
    if cls_name == "VGGishBinary":
        return CNNGradCAM(model)
    if cls_name == "ASTBinary":
        return TransformerGradCAM(model)
    raise ValueError(f"No Grad-CAM variant registered for {cls_name}")
