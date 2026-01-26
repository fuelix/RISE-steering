"""
Householder Pseudo-Rotation (HPR) baseline.

Implementation of Householder Pseudo-Rotation from Chai et al. (2024)
"Language-Universal Semantic Transformations via Pseudo-Rotations".

HPR uses a LIMITED number of Householder reflections to construct
pseudo-rotation matrices that transform embeddings while preserving norms.
Unlike full Procrustes (which finds the optimal orthogonal matrix), HPR
is constrained to k reflections, making it less expressive but more
interpretable and efficient.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import SteeringMethod, SteeringResult

logger = logging.getLogger(__name__)


class HPRMethod(SteeringMethod):
    """
    Householder Pseudo-Rotation (HPR) steering method.

    HPR learns a transformation using a LIMITED number of Householder
    reflections. This is intentionally less expressive than full Procrustes
    to match the method described in the original paper.

    Key characteristics:
    - Preserves embedding norms (orthogonal transformation)
    - Limited to k Householder reflections (typically k=2)
    - Less expressive than full orthogonal matrix by design

    The transformation is: H_k @ H_{k-1} @ ... @ H_1 @ x
    where each H_i = I - 2 * v_i * v_i^T is a Householder reflection.

    Reference:
        Chai et al. (2024). "Language-Universal Semantic Transformations
        via Pseudo-Rotations." ACL 2024.

    Attributes:
        n_reflections: Number of Householder reflections to use (default: 2).
        householder_vectors: Learned Householder vectors, shape [k, d].
    """

    def __init__(self, n_reflections: int = 2):
        """
        Initialize HPR method.

        Args:
            n_reflections: Number of Householder reflections to compose.
                More reflections = more expressive but the method is
                intentionally limited. Typical values: 1-4.
        """
        super().__init__(name="HPR")
        self.n_reflections = n_reflections
        self.householder_vectors: Optional[torch.Tensor] = None

    def _compute_householder_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Householder reflection matrix from a unit vector.

        H = I - 2 * v * v^T

        Args:
            v: Unit Householder vector, shape [d].

        Returns:
            Householder matrix, shape [d, d].
        """
        d = v.shape[0]
        H = torch.eye(d, device=v.device, dtype=v.dtype) - 2.0 * torch.outer(v, v)
        return H

    def _apply_householder_sequence(
        self, x: torch.Tensor, vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a sequence of Householder reflections to input.

        Computes H_k @ H_{k-1} @ ... @ H_1 @ x efficiently without
        forming the full matrices.

        Args:
            x: Input tensor, shape [d] or [N, d].
            vectors: Householder vectors, shape [k, d].

        Returns:
            Transformed tensor, same shape as x.
        """
        result = x.clone()
        for i in range(vectors.shape[0]):
            v = vectors[i]
            # H @ x = x - 2 * v * (v^T @ x)
            if result.dim() == 1:
                result = result - 2.0 * v * torch.dot(v, result)
            else:
                # Batched: [N, d] - 2 * [d] * [N, 1]
                dots = (result @ v).unsqueeze(1)  # [N, 1]
                result = result - 2.0 * dots * v.unsqueeze(0)  # [N, d]
        return result

    def _greedy_householder_fitting(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        n_reflections: int,
    ) -> torch.Tensor:
        """
        Greedily find Householder vectors to minimize reconstruction error.

        At each step, find the Householder vector that best reduces the
        error between the current transformed X and target Y.

        Args:
            X: Source embeddings, [N, d].
            Y: Target embeddings, [N, d].
            n_reflections: Number of Householder vectors to find.

        Returns:
            Householder vectors, shape [k, d].
        """
        N, d = X.shape
        device, dtype = X.device, X.dtype

        vectors = []
        current_X = X.clone()

        for k in range(n_reflections):
            # Find the Householder vector that minimizes ||H @ current_X - Y||
            # For a single Householder reflection H = I - 2vv^T:
            # H @ X = X - 2 * v * (v^T @ X)
            #
            # We want to find v that minimizes:
            # ||X - 2 * v * (X @ v) - Y||^2
            #
            # Let R = X - Y (residual). We want to minimize:
            # ||R - 2 * v * (X @ v)||^2
            #
            # This is solved by finding the direction that maximally
            # correlates the residual with the projection onto X.

            R = current_X - Y  # [N, d] residual

            # Compute the matrix M = X^T @ R
            # The optimal v is related to the leading singular vector
            M = current_X.T @ R  # [d, d]

            # Use SVD to find the direction of maximum variance
            # We want the direction that captures most of the residual
            U, S, Vt = torch.linalg.svd(M)

            # Take the leading left singular vector as Householder direction
            v = U[:, 0]
            v = F.normalize(v, dim=0)

            vectors.append(v)

            # Apply this reflection to current_X for next iteration
            current_X = self._apply_householder_sequence(
                current_X, v.unsqueeze(0)
            )

        return torch.stack(vectors, dim=0)

    def fit(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Learn the HPR transformation from training pairs.

        Uses greedy algorithm to find k Householder vectors that
        minimize reconstruction error, subject to the constraint
        of using only k reflections.

        Args:
            neutral_embeddings: Source embeddings, [N, d].
            transformed_embeddings: Target embeddings, [N, d].

        Returns:
            Training statistics.
        """
        if neutral_embeddings.shape != transformed_embeddings.shape:
            raise ValueError("Shape mismatch between embeddings")

        N, d = neutral_embeddings.shape
        logger.info(
            f"HPR: Learning transformation from {N} pairs "
            f"with {self.n_reflections} reflections"
        )

        # Normalize embeddings
        X = F.normalize(neutral_embeddings, dim=1)
        Y = F.normalize(transformed_embeddings, dim=1)

        # Find Householder vectors greedily
        self.householder_vectors = self._greedy_householder_fitting(
            X, Y, self.n_reflections
        )

        self._is_fitted = True

        # Compute reconstruction error
        Y_pred = self._apply_householder_sequence(X, self.householder_vectors)
        recon_error = torch.mean(torch.norm(Y - Y_pred, dim=1)).item()

        # Also compute what full Procrustes would achieve for comparison
        M = X.T @ Y
        U, S, Vt = torch.linalg.svd(M)
        R_optimal = U @ Vt
        Y_optimal = X @ R_optimal
        optimal_error = torch.mean(torch.norm(Y - Y_optimal, dim=1)).item()

        logger.info(
            f"HPR: Learned with {self.n_reflections} reflections, "
            f"recon_error = {recon_error:.4f} "
            f"(optimal Procrustes = {optimal_error:.4f})"
        )

        return {
            "num_pairs": N,
            "n_reflections": self.n_reflections,
            "reconstruction_error": recon_error,
            "optimal_procrustes_error": optimal_error,
            "expressiveness_gap": recon_error - optimal_error,
        }

    def transform(
        self,
        embedding: torch.Tensor,
        normalize_output: bool = True,
        **kwargs,
    ) -> SteeringResult:
        """
        Apply HPR transformation to an embedding.

        Args:
            embedding: Input embedding, shape [d].
            normalize_output: Whether to normalize the result.

        Returns:
            SteeringResult with transformed embedding.
        """
        if not self._is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")

        vectors = self.householder_vectors.to(embedding.device, embedding.dtype)
        transformed = self._apply_householder_sequence(embedding, vectors)

        if normalize_output:
            transformed = F.normalize(transformed, dim=0)

        return SteeringResult(
            method_name=self.name,
            predicted_embedding=transformed,
            source_embedding=embedding,
            metadata={"n_reflections": self.n_reflections},
        )

    def save(self, path: Path) -> None:
        """Save the learned transformation."""
        if not self._is_fitted:
            raise ValueError("Nothing to save - method not fitted")

        state = {
            "householder_vectors": self.householder_vectors.cpu(),
            "n_reflections": self.n_reflections,
        }
        torch.save(state, path, weights_only=False)
        logger.info(f"HPR method saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "HPRMethod":
        """Load a saved HPR method."""
        state = torch.load(path, weights_only=True)
        method = cls(n_reflections=state["n_reflections"])
        method.householder_vectors = state["householder_vectors"]
        method._is_fitted = True
        logger.info(f"HPR method loaded from {path}")
        return method

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            "name": self.name,
            "n_reflections": self.n_reflections,
            "vectors_shape": (
                list(self.householder_vectors.shape)
                if self.householder_vectors is not None
                else None
            ),
        }
