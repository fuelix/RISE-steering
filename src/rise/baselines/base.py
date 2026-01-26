"""
Abstract base class for steering methods.

This module defines the common interface that all steering methods
(RISE, Park, CAA, HPR) must implement for fair comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SteeringResult:
    """Result from applying a steering method."""

    method_name: str
    """Name of the steering method."""

    predicted_embedding: torch.Tensor
    """The predicted/steered embedding."""

    source_embedding: torch.Tensor
    """The original input embedding."""

    confidence: Optional[float] = None
    """Optional confidence score for the prediction."""

    metadata: Optional[Dict[str, Any]] = None
    """Additional method-specific metadata."""


class SteeringMethod(ABC):
    """
    Abstract base class for semantic steering methods.

    All steering methods must implement:
    - fit(): Learn the steering transformation from training data
    - transform(): Apply the learned transformation to new inputs
    - save()/load(): Persist and restore the learned model

    This enables fair comparison across different approaches.
    """

    def __init__(self, name: str):
        """
        Initialize the steering method.

        Args:
            name: Human-readable name for the method.
        """
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Learn the steering transformation from training data.

        Args:
            neutral_embeddings: Original/neutral embeddings, shape [N, d].
            transformed_embeddings: Transformed embeddings, shape [N, d].
            **kwargs: Method-specific parameters.

        Returns:
            Dictionary with training statistics.
        """
        pass

    @abstractmethod
    def transform(
        self,
        embedding: torch.Tensor,
        **kwargs,
    ) -> SteeringResult:
        """
        Apply the learned transformation to a new embedding.

        Args:
            embedding: Input embedding to transform, shape [d].
            **kwargs: Method-specific parameters.

        Returns:
            SteeringResult with the transformed embedding.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the learned model to disk.

        Args:
            path: Path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "SteeringMethod":
        """
        Load a saved model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded SteeringMethod instance.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Whether the method has been fitted."""
        return self._is_fitted

    def get_config(self) -> Dict[str, Any]:
        """
        Get the method's configuration.

        Returns:
            Dictionary of configuration parameters.
        """
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
