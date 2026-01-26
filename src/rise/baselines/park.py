"""
Park's Linear Representation Method baseline.

Implementation of Park et al. (2024) "The Linear Representation Hypothesis
and the Geometry of Large Language Models".

This method computes a concept direction as the mean difference between
positive and negative examples, then applies linear steering.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .base import SteeringMethod, SteeringResult

logger = logging.getLogger(__name__)


class ParkMethod(SteeringMethod):
    """
    Park's Linear Representation steering method.

    This method learns a concept direction by computing the mean difference
    between positive (transformed) and negative (neutral) embeddings:

        direction = mean(positive) - mean(negative)

    Steering is then applied by adding a scaled version of this direction:

        steered = embedding + alpha * direction

    Reference:
        Park et al. (2024). "The Linear Representation Hypothesis and
        the Geometry of Large Language Models." ICML 2024.

    Attributes:
        alpha: Steering strength (default 0.4 per paper).
        direction: Learned concept direction.
    """

    def __init__(self, alpha: float = 0.4):
        """
        Initialize Park's method.

        Args:
            alpha: Steering strength. Paper recommends 0.4.
        """
        super().__init__(name="Park")
        self.alpha = alpha
        self.direction: Optional[torch.Tensor] = None

    def fit(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
        normalize_direction: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Learn the concept direction from training pairs.

        Args:
            neutral_embeddings: Neutral embeddings (negative class), [N, d].
            transformed_embeddings: Transformed embeddings (positive class), [N, d].
            normalize_direction: Whether to normalize the direction vector.

        Returns:
            Training statistics.
        """
        if neutral_embeddings.shape != transformed_embeddings.shape:
            raise ValueError("Shape mismatch between neutral and transformed embeddings")

        N, d = neutral_embeddings.shape
        logger.info(f"Park: Learning concept direction from {N} pairs")

        # Compute mean difference
        pos_mean = transformed_embeddings.mean(dim=0)
        neg_mean = neutral_embeddings.mean(dim=0)
        direction = pos_mean - neg_mean

        if normalize_direction:
            direction = F.normalize(direction, dim=0)

        self.direction = direction
        self._is_fitted = True

        direction_norm = torch.norm(direction).item()
        logger.info(f"Park: Direction learned, ||d|| = {direction_norm:.4f}")

        return {
            "num_pairs": N,
            "direction_norm": direction_norm,
            "alpha": self.alpha,
        }

    def transform(
        self,
        embedding: torch.Tensor,
        alpha: Optional[float] = None,
        normalize_output: bool = True,
        **kwargs,
    ) -> SteeringResult:
        """
        Apply Park's steering to an embedding.

        Args:
            embedding: Input embedding, shape [d].
            alpha: Steering strength (overrides default).
            normalize_output: Whether to normalize the result.

        Returns:
            SteeringResult with steered embedding.
        """
        if not self._is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")

        alpha = alpha if alpha is not None else self.alpha

        # Apply steering: steered = embedding + alpha * direction
        direction = self.direction.to(embedding.device, embedding.dtype)
        steered = embedding + alpha * direction

        if normalize_output:
            steered = F.normalize(steered, dim=0)

        return SteeringResult(
            method_name=self.name,
            predicted_embedding=steered,
            source_embedding=embedding,
            metadata={"alpha": alpha},
        )

    def save(self, path: Path) -> None:
        """Save the learned direction."""
        if not self._is_fitted:
            raise ValueError("Nothing to save - method not fitted")

        state = {
            "direction": self.direction.cpu(),
            "alpha": self.alpha,
        }
        torch.save(state, path)
        logger.info(f"Park method saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ParkMethod":
        """Load a saved Park method."""
        state = torch.load(path, weights_only=True)
        method = cls(alpha=state["alpha"])
        method.direction = state["direction"]
        method._is_fitted = True
        logger.info(f"Park method loaded from {path}")
        return method

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            "name": self.name,
            "alpha": self.alpha,
            "direction_dim": len(self.direction) if self.direction is not None else None,
        }
