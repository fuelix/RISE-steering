"""
Type definitions for RISE.

This module defines common types used throughout the RISE package,
including result containers and type aliases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch


# =============================================================================
# Type Aliases
# =============================================================================

# A tensor representing a point on the unit hypersphere S^{d-1}
UnitVector = torch.Tensor

# A tensor representing a tangent vector at some base point
TangentVector = torch.Tensor

# A rotation/reflection matrix (orthogonal matrix)
RotorMatrix = torch.Tensor


# =============================================================================
# Result Containers
# =============================================================================

@dataclass
class RISEPrototypeResult:
    """Result of learning a RISE prototype."""

    prototype: torch.Tensor
    """The learned prototype vector in tangent space at e1 (if canonicalized)."""

    num_pairs: int
    """Number of training pairs used."""

    canonicalized: bool
    """Whether canonicalization was applied during learning."""

    prototype_norm: float
    """L2 norm of the prototype vector."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the learning process."""


@dataclass
class TransformPrediction:
    """Result of predicting a transformation using RISE."""

    predicted_embedding: torch.Tensor
    """The predicted transformed embedding."""

    source_embedding: torch.Tensor
    """The original source embedding."""

    cosine_similarity: float
    """Cosine similarity between source and predicted."""


@dataclass
class AlignmentScore:
    """Alignment score between predicted and actual transformations."""

    score: float
    """Mean cosine similarity between predictions and ground truth."""

    std: float
    """Standard deviation of alignment scores."""

    num_samples: int
    """Number of samples evaluated."""

    per_sample_scores: Optional[List[float]] = None
    """Individual scores per sample (if requested)."""


@dataclass
class CrossLanguageTransferResult:
    """Result of cross-language transfer evaluation."""

    source_language: str
    """Language the prototype was trained on."""

    target_language: str
    """Language the prototype was evaluated on."""

    alignment_score: float
    """Alignment score on the target language."""

    transformation_type: str
    """Type of transformation (negation, politeness, conditionality)."""
