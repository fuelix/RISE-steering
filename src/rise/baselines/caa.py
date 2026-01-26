"""
Concept Activation Addition (CAA) baseline.

Implementation of Contrastive Activation Addition from Rimsky et al. (2023)
"Steering Llama 2 via Contrastive Activation Addition".

CAA computes a steering vector as the mean difference of activations
between positive and negative examples, similar to Park but with
different application strategies.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .base import SteeringMethod, SteeringResult

logger = logging.getLogger(__name__)


class CAAMethod(SteeringMethod):
    """
    Concept Activation Addition (CAA) steering method.

    CAA learns a steering vector from contrastive examples and applies
    it through addition at specific positions/layers.

    Key differences from Park:
    - CAA is typically applied to hidden states at specific layers
    - Can use position-specific steering (e.g., last token only)
    - Often uses higher steering strengths

    Reference:
        Rimsky et al. (2023). "Steering Llama 2 via Contrastive
        Activation Addition." arXiv:2312.06681.

    Attributes:
        strength: Steering strength (default 2.0 per typical usage).
        steering_vector: Learned steering vector.
    """

    def __init__(self, strength: float = 2.0):
        """
        Initialize CAA method.

        Args:
            strength: Steering strength. Typical values: 1.0-3.0.
        """
        super().__init__(name="CAA")
        self.strength = strength
        self.steering_vector: Optional[torch.Tensor] = None

    def fit(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
        normalize_vector: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute the CAA steering vector.

        Args:
            neutral_embeddings: Negative class embeddings, [N, d].
            transformed_embeddings: Positive class embeddings, [N, d].
            normalize_vector: Whether to normalize the steering vector.

        Returns:
            Training statistics.
        """
        if neutral_embeddings.shape != transformed_embeddings.shape:
            raise ValueError("Shape mismatch between embeddings")

        N, d = neutral_embeddings.shape
        logger.info(f"CAA: Computing steering vector from {N} pairs")

        # CAA vector = mean(positive) - mean(negative)
        pos_mean = transformed_embeddings.mean(dim=0)
        neg_mean = neutral_embeddings.mean(dim=0)
        steering_vector = pos_mean - neg_mean

        if normalize_vector:
            steering_vector = F.normalize(steering_vector, dim=0)

        self.steering_vector = steering_vector
        self._is_fitted = True

        vector_norm = torch.norm(steering_vector).item()
        logger.info(f"CAA: Steering vector computed, ||v|| = {vector_norm:.4f}")

        return {
            "num_pairs": N,
            "vector_norm": vector_norm,
            "strength": self.strength,
        }

    def transform(
        self,
        embedding: torch.Tensor,
        strength: Optional[float] = None,
        normalize_output: bool = True,
        **kwargs,
    ) -> SteeringResult:
        """
        Apply CAA steering to an embedding.

        Args:
            embedding: Input embedding, shape [d].
            strength: Steering strength (overrides default).
            normalize_output: Whether to normalize the result.

        Returns:
            SteeringResult with steered embedding.
        """
        if not self._is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")

        strength = strength if strength is not None else self.strength

        # Apply steering: steered = embedding + strength * vector
        vector = self.steering_vector.to(embedding.device, embedding.dtype)
        steered = embedding + strength * vector

        if normalize_output:
            steered = F.normalize(steered, dim=0)

        return SteeringResult(
            method_name=self.name,
            predicted_embedding=steered,
            source_embedding=embedding,
            metadata={"strength": strength},
        )

    def save(self, path: Path) -> None:
        """Save the learned steering vector."""
        if not self._is_fitted:
            raise ValueError("Nothing to save - method not fitted")

        state = {
            "steering_vector": self.steering_vector.cpu(),
            "strength": self.strength,
        }
        torch.save(state, path)
        logger.info(f"CAA method saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CAAMethod":
        """Load a saved CAA method."""
        state = torch.load(path, weights_only=True)
        method = cls(strength=state["strength"])
        method.steering_vector = state["steering_vector"]
        method._is_fitted = True
        logger.info(f"CAA method loaded from {path}")
        return method

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            "name": self.name,
            "strength": self.strength,
            "vector_dim": len(self.steering_vector) if self.steering_vector is not None else None,
        }
