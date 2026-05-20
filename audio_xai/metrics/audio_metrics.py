from __future__ import annotations  # <-- Add this at the absolute top
import torch
import numpy as np

from functools import partial

from scipy.spatial.distance import cdist
from tqdm import tqdm
import logging


def ensure_tensor(x: np.ndarray | torch.Tensor, device=None) -> torch.Tensor:
    """Ensures input is a PyTorch tensor on the correct device."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(device, non_blocking=True) if device else x


def ensure_ndarray(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Ensures input is a NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x


def nearest_neighbour_distances(input_features: torch.Tensor, nearest_k: int) -> torch.Tensor:
    """Computes distances to the kth nearest neighbours."""
    distances = torch.cdist(input_features, input_features)
    # k=nearest_k + 1 because the 1st closest neighbor is always the point itself (distance=0)
    radii = torch.kthvalue(distances, k=nearest_k + 1, dim=-1)[0]
    return radii


def prdc(reference: AudioMetricsData, candidate: AudioMetricsData, nearest_k: int) -> dict:
    """Computes precision, recall, density, and coverage given two manifolds.
    
    Accepts AudioMetricsData instances directly and extracts radii and embeddings natively.
    """
    if reference.embeddings is None or candidate.embeddings is None:
        raise ValueError("PRDC calculation requires stored embeddings in both datasets.")

    ref_nn_radii = reference.get_radii(nearest_k)
    cand_nn_radii = candidate.get_radii(nearest_k)

    distance_ref_cand = torch.cdist(reference.embeddings, candidate.embeddings)

    # Calculate metrics matching PRDC definitions
    precision = (distance_ref_cand < ref_nn_radii[:, None]).any(dim=0).double().mean().item()
    recall = (distance_ref_cand < cand_nn_radii[None, :]).any(dim=1).double().mean().item()
    
    density = (1.0 / float(nearest_k)) * (
        (distance_ref_cand < ref_nn_radii[:, None]).sum(dim=0).double().mean().item()
    )
    
    coverage = (distance_ref_cand.min(dim=1)[0] < ref_nn_radii).double().mean().item()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage)

class AudioMetricsData:
    def __init__(self, store_embeddings: bool = True):
        self.mean = None
        self.cov = None
        self.n = 0
        self.store_embeddings = store_embeddings
        self.embeddings = None
        self.radii = {}
        self.dtype = torch.float64

    def serialize(self) -> dict:
        """Serializes current state into a dictionary of deep-copy clean types."""
        return {
            "mean": self.mean.clone() if isinstance(self.mean, torch.Tensor) else None,
            "cov": self.cov.clone() if isinstance(self.cov, torch.Tensor) else None,
            "n": self.n,
            "store_embeddings": self.store_embeddings,
            "embeddings": self.embeddings.clone() if isinstance(self.embeddings, torch.Tensor) else None,
            "radii": {k: v.clone() for k, v in self.radii.items()},
            "dtype": str(self.dtype)
        }

    @classmethod
    def deserialize(cls, state: dict):
        """Reconstructs the object from a serialized dictionary."""
        self = cls()
        self.n = state["n"]
        self.store_embeddings = state["store_embeddings"]
        self.mean = state["mean"]
        self.cov = state["cov"]
        self.embeddings = state["embeddings"]
        self.radii = state["radii"]
        
        # Re-resolve dtype string back to actual torch property
        dtype_str = state.get("dtype", "torch.float64")
        self.dtype = torch.float64 if "float64" in dtype_str else torch.float32
        return self

    def add(self, embeddings: torch.Tensor | np.ndarray):
        """Validates, converts, and streams batch embeddings into metric statistics."""
        embeddings = ensure_tensor(embeddings).to(dtype=self.dtype)
        if embeddings.ndim ==  1:
            embeddings = embeddings.unsqueeze(0)

        n_new = len(embeddings)
        mean_new = torch.mean(embeddings, dim=0)
        
        if n_new == 1:
            d = embeddings.shape[-1]
            cov_new = torch.zeros((d, d), dtype=self.dtype, device=embeddings.device)
        else:
            cov_new = torch.cov(embeddings.T)

        self._update_stats(mean_new, cov_new, n_new)
        
        if self.store_embeddings:
            self._update_embeddings(embeddings)

    def recompute_stats(self):
        """Forces statistical recomputations if embeddings are cached."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            self.n = len(self.embeddings)
            self.mean = torch.mean(self.embeddings, dim=0)
            if self.n == 1:
                d = self.embeddings.shape[-1]
                self.cov = torch.zeros((d, d), dtype=self.dtype, device=self.embeddings.device)
            else:
                self.cov = torch.cov(self.embeddings.T)

    def get_radii(self, k_neighbor: int) -> torch.Tensor:
        """Retrieves or calculates memoized k-NN radii boundaries."""
        key = f"radii_{k_neighbor}"
        radii = self.radii.get(key)
        if radii is None and self.embeddings is not None:
            radii = nearest_neighbour_distances(self.embeddings, k_neighbor)
            self.radii[key] = radii
        elif radii is None:
            raise ValueError("Embeddings were not stored. Cannot calculate k-NN radii.")
        return radii

    def _update_embeddings(self, embeddings: torch.Tensor):
        if self.embeddings is None:
            self.embeddings = embeddings.clone()
        else:
            self.embeddings = torch.cat((self.embeddings, embeddings), dim=0)
        # Invalidate cached radii since underlying manifolds expanded
        self.radii.clear()

    def __len__(self) -> int:
        return self.n

    def _update_stats(self, mean: torch.Tensor, cov: torch.Tensor, n: int):
        if self.n == 0 or self.mean is None:
            self.mean = mean
            self.cov = cov
            self.n = n
            return
            
        n_prod = self.n * n
        n_total = self.n + n
        
        new_mean = (self.n * self.mean + n * mean) / n_total
        diff_mean = self.mean - mean
        diff_mean_mat = torch.einsum("i,j->ij", diff_mean, diff_mean)
        
        w_self = (self.n - 1) / (n_total - 1) if n_total > 1 else 0.0
        w_other = (n - 1) / (n_total - 1) if n_total > 1 else 0.0
        w_diff = (n_prod / n_total) / (n_total - 1) if n_total > 1 else 0.0
        
        self.cov = (w_self * self.cov) + (w_other * cov) + (w_diff * diff_mean_mat)
        self.mean = new_mean
        self.n = n_total

    def __iadd__(self, other):
        assert isinstance(other, AudioMetricsData)
        if other.n == 0:
            return self
        if self.n == 0:
            self.store_embeddings = other.store_embeddings
            self.dtype = other.dtype
            
        assert self.store_embeddings == other.store_embeddings
        self._update_stats(other.mean, other.cov, other.n)
        if self.store_embeddings and other.embeddings is not None:
            self._update_embeddings(other.embeddings)
        return self

    def __add__(self, other):
        new = AudioMetricsData(store_embeddings=self.store_embeddings)
        new += self
        new += other
        return new
