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
        """
        Initialize Grad-CAM scaffolding and attach forward/backward hooks to the model's target layer.
        
        Stores the provided audio classification model and initializes internal bookkeeping for captured
        activations, gradients, and hook handles, then registers hooks on model.target_layer to capture
        layer outputs and their gradients during forward/backward passes.
        
        Parameters:
            model (AudioClassifier): The model to inspect; must expose a `target_layer` attribute on which
                forward and backward hooks will be registered.
        """
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Attach forward and full-backward hooks to the model's target layer to capture its activations and the gradients w.r.t. its outputs.
        
        The forward hook saves the layer's forward output to self._activations. The backward hook saves the first element of grad_out (the gradient w.r.t. the layer output) to self._gradients. Both hook handles are appended to self._handles; the backward hook is registered with register_full_backward_hook to support higher-order gradients.
        """
        layer = self.model.target_layer

        def fwd_hook(_module, _inputs, output):
            """
            Store the forward hook's output into self._activations.
            
            Parameters:
                _module: The layer module that produced the output (unused).
                _inputs: The inputs passed to the layer during forward (unused).
                output: The forward hook output to capture and store.
            """
            self._activations = output

        def bwd_hook(_module, _grad_in, grad_out):
            # grad_out is a tuple; first element is the gradient w.r.t. output.
            """
            Capture the gradient tensor for the target layer's output from a backward hook and store it on self._gradients.
            
            Parameters:
                _module: The layer module receiving the backward hook (unused).
                _grad_in: Tuple of gradients flowing into the layer from earlier backward calls (unused).
                grad_out: Tuple of gradients flowing out of the layer; the first element is the gradient with respect to the layer's output and is saved.
            """
            self._gradients = grad_out[0]

        self._handles.append(layer.register_forward_hook(fwd_hook))
        # full_backward_hook works with create_graph=True; the older
        # register_backward_hook does not.
        self._handles.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        """
        Remove all registered hook handles from the model and clear the internal handle list.
        
        This detaches any forward/backward hooks previously registered by the instance so they no longer receive callbacks, and empties self._handles.
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        """
        Enter the context manager and return the Grad-CAM instance.
        
        Returns:
            self: The Grad-CAM instance to be used within the context.
        """
        return self

    def __exit__(self, *_):
        """
        Remove all registered forward/backward hooks and clear internal handles when exiting the context manager.
        """
        self.remove_hooks()

    @abstractmethod
    def _build_heatmap(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor: """
Construct a 2D per-sample heatmap from stored layer activations and their gradients.

Parameters:
    activations (torch.Tensor): Activation tensor captured from the model's target layer for the current batch.
        Implementations may expect shapes such as `[B, C, H, W]` (CNN) or `[B, N_tokens, D]` (transformer tokens).
    gradients (torch.Tensor): Gradients of the target score with respect to `activations`, with the same shape as `activations`.

Returns:
    torch.Tensor: A heatmap tensor of shape `[B, H, W]` (or `[B, F, T]` for transformer-derived grids) representing per-sample importance scores.
"""
...

    def __call__(
        self,
        waveform: torch.Tensor,
        target_class: torch.Tensor | int | None = None,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute a Grad-CAM heatmap for each input waveform.
        
        Parameters:
            waveform (torch.Tensor): Input batch of waveforms with shape [B, T].
            target_class (torch.Tensor | int | None): Target class indices to explain. If None, the per-sample argmax of the model logits is used. If an int is provided it is expanded to all batch elements.
            create_graph (bool): If True, the returned heatmap is differentiable with respect to the input (useful when the heatmap is part of a loss you will backpropagate through).
        
        Returns:
            torch.Tensor: Heatmap tensor with shape [B, H, W] (per-sample spatial/spectrogram heatmaps).
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
        """
        Compute a spatial Grad-CAM heatmap from convolutional activations and their gradients.
        
        Parameters:
            activations (torch.Tensor): Activation tensor from a convolutional layer with shape [B, C, H, W].
            gradients (torch.Tensor): Gradients w.r.t. `activations` with shape [B, C, H, W].
        
        Returns:
            torch.Tensor: ReLU-applied class activation map with shape [B, H, W], produced by channel-weighted summation where channel weights are the spatial global average of `gradients`.
        """
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
        """
        Initialize a TransformerGradCAM configured for transformer token activations.
        
        Parameters:
            model (AudioClassifier): The model whose target layer hooks will be used to capture activations and gradients.
            num_special_tokens (int): Number of leading special tokens to exclude from token-to-grid mapping (e.g., [CLS], [SEP]).
            freq_patches (int): Number of patch rows (frequency dimension) expected when reshaping token sequence into a 2D grid.
            time_patches (int): Number of patch columns (time dimension) expected when reshaping token sequence into a 2D grid.
        """
        super().__init__(model)
        self.num_special = num_special_tokens
        self.freq_patches = freq_patches
        self.time_patches = time_patches

    def _build_heatmap(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        # Drop CLS / distillation tokens.
        """
        Construct a spectrogram-shaped Grad-CAM heatmap from transformer token activations and their gradients.
        
        Parameters:
            activations (torch.Tensor): Token activations with shape [B, N_tokens, D]; special tokens are expected at the start.
            gradients (torch.Tensor): Gradients w.r.t. the token activations with the same shape as `activations`.
        
        Description:
            Drops the first `self.num_special` tokens, optionally truncates tokens to match a grid inferred from
            `self.freq_patches` if the token count differs, reshapes the remaining tokens into a
            [B, freq_patches, time_patches, D] grid, computes per-dimension weights by averaging gradients
            over the grid, and produces a non-negative heatmap by summing weighted activations across the
            embedding dimension and applying ReLU.
        
        Returns:
            torch.Tensor: Heatmap of shape [B, freq_patches, time_patches] representing importance in the
            frequency-time spectrogram grid.
        """
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
    """
    Selects the appropriate Grad-CAM implementation for the supplied audio model.
    
    Parameters:
        model (AudioClassifier): The model instance used to determine which Grad-CAM variant to construct.
    
    Returns:
        GradCAMBase: An instance of the matching Grad-CAM subclass for the model.
    
    Raises:
        ValueError: If no Grad-CAM variant is registered for the model's class name.
    """
    cls_name = type(model).__name__
    if cls_name == "VGGishBinary":
        return CNNGradCAM(model)
    if cls_name == "ASTBinary":
        return TransformerGradCAM(model)
    raise ValueError(f"No Grad-CAM variant registered for {cls_name}")
