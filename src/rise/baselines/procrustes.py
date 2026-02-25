"""
Orthogonal Procrustes baseline.

Finds the optimal orthogonal matrix W that minimizes ||neutral @ W - transformed||_F
via SVD of neutral^T @ transformed.

This baseline learns a global linear (orthogonal) mapping but does not account
for the local Riemannian structure of the hypersphere.
"""

import torch
import torch.nn.functional as F

from ..utils.types import TransformPrediction


class Procrustes:
    """
    Orthogonal Procrustes alignment baseline for semantic transformations.

    Solves: W* = argmin_W ||neutral @ W - transformed||_F  s.t. W^T W = I
    Solution: W = U V^T where M = neutral^T @ transformed = U S V^T
    Prediction for column vector x: W^T @ x  (equivalently, x @ W for row vectors)

    Implements the same duck-typed interface as RISE:
        .fit(neutral_embeddings, transformed_embeddings)
        .transform(embedding=tensor) -> TransformPrediction
    """

    def __init__(self):
        self.W: torch.Tensor | None = None
        self._is_fitted = False

    def fit(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
    ) -> None:
        """Learn the optimal orthogonal mapping from paired embeddings."""
        neutral_embeddings = F.normalize(neutral_embeddings, dim=1)
        transformed_embeddings = F.normalize(transformed_embeddings, dim=1)

        # M = neutral^T @ transformed  (d x d)
        M = neutral_embeddings.T @ transformed_embeddings
        U, _, Vt = torch.linalg.svd(M)
        self.W = U @ Vt
        self._is_fitted = True

    def transform(
        self,
        embedding: torch.Tensor,
        text: str | None = None,
    ) -> TransformPrediction:
        """Predict the transformed embedding via orthogonal mapping."""
        if not self._is_fitted:
            raise ValueError("Procrustes not fitted. Call fit() first.")

        embedding = F.normalize(embedding, dim=0)
        # W solves min ||N W - T||_F (row-vector convention), so for a
        # column vector x the prediction is W^T x.
        predicted = F.normalize(self.W.T @ embedding, dim=0)
        cosine_sim = F.cosine_similarity(embedding, predicted, dim=0).item()

        return TransformPrediction(
            predicted_embedding=predicted,
            source_embedding=embedding,
            cosine_similarity=cosine_sim,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
