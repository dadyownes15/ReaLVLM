from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from joblib import dump, load

# Import heavy libs lazily to avoid unnecessary overhead if a user only needs one reducer.
from sklearn.decomposition import PCA as _SKPCA  # type: ignore

try:
    import umap as umap_learn  # type: ignore
except ImportError:  # pragma: no cover – handled at runtime
    umap_learn = None  # type: ignore


class DimensionReduction:
    """Base class – defines I/O & persistence logic."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.model = None  # Will be set by subclasses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reduce_dimensions(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:  # noqa: D401, N803
        """Reduce dimension – to be implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _emb_list_to_ndarray(self, embeddings: List[torch.Tensor]) -> np.ndarray:
        """Stack list of 1‑D *torch* tensors into a (n_samples, dim) ndarray."""
        if not embeddings:
            raise ValueError("`embeddings` list is empty – nothing to transform.")
        return np.vstack([e.detach().cpu().numpy() for e in embeddings])

    def _ndarray_to_emb_list(self, arr: np.ndarray) -> List[torch.Tensor]:
        """Convert ndarray back to list of 1‑D *torch* tensors (float32)."""
        return [torch.from_numpy(row).float() for row in arr]

    def _load_or_fit(self, X: np.ndarray):
        """Load reducer from disk if present; otherwise fit & persist."""
        self._fit(X)

    # Each subclass must implement _fit(X) & _transform(X).

    def fit(self, X: List[torch.Tensor]):  # pragma: no cover
        raise NotImplementedError

    def _transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class PCA(DimensionReduction):
    """Principal Component Analysis reducer (linear, fast)."""

    def __init__(
        self,
        n_components: int = 64,
        svd_solver: str = "auto",
        whiten: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(n_components)
        # Pre‑configure the PCA instance – will be created in _fit()
        self._cfg = {
            "n_components": n_components,
            "svd_solver": svd_solver,
            "whiten": whiten,
            "random_state": random_state,
        }

    # ------------------------------------------------------------------
    # DimensionReduction overrides
    # ------------------------------------------------------------------

    def reduce_dimensions(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.model is None:
            raise RuntimeError("PCA model is not loaded/fitted yet.")
        X = self._emb_list_to_ndarray(embeddings)
        reduced = self._transform(X)
        return self._ndarray_to_emb_list(reduced)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def fit(self, X: List[torch.Tensor]):
        self.model = _SKPCA(**self._cfg)
        X_np = self._emb_list_to_ndarray(X)
        self.model.fit(X_np)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA model is not loaded/fitted yet.")
        return self.model.transform(X).astype(np.float32)


class UMAP(DimensionReduction):
    """Uniform Manifold Approximation and Projection reducer (non-linear)."""

    def __init__(
        self,
        n_components: int = 16,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_epochs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(n_components)
        if umap_learn is None:
            raise ImportError(
                "umap-learn is not installed. Please install it e.g. with `pip install umap-learn`."
            )
        # Store UMAP-specific configuration
        self._cfg = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "n_epochs": n_epochs,
            "random_state": random_state,
            "verbose": verbose,
        }

    # ------------------------------------------------------------------
    # DimensionReduction overrides
    # ------------------------------------------------------------------

    def reduce_dimensions(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Reduce dimension of embeddings using the fitted UMAP model."""
        if self.model is None:
            raise RuntimeError("UMAP model is not fitted yet. Call .fit() first.")
        X_np = self._emb_list_to_ndarray(embeddings)
        reduced_X_np = self._transform(X_np)
        return self._ndarray_to_emb_list(reduced_X_np)

    # ------------------------------------------------------------------
    # Internal helpers (fitting and transforming)
    # ------------------------------------------------------------------

    def fit(self, embeddings: List[torch.Tensor]):
        """Fit the UMAP model to the provided embeddings."""
        # __init__ already checks for umap_learn availability.
        # Instantiate UMAP model with configuration.
        # umap_learn.UMAP will use its own default for n_epochs if self._cfg["n_epochs"] is None.
        self.model = umap_learn.UMAP(**self._cfg)

        # Convert embeddings to NumPy array
        X_np = self._emb_list_to_ndarray(embeddings)

        # Fit the model
        self.model.fit(X_np)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted UMAP model."""
        if self.model is None:
            raise RuntimeError("UMAP model is not fitted yet.")
        
        transformed_X = self.model.transform(X)
        return transformed_X.astype(np.float32)

