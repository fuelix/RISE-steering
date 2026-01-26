"""
Utility modules for RISE.

Contains constants, type definitions, and helper functions.
"""

from .constants import (
    ARCCOS_CLAMP_EPS,
    NEAR_IDENTITY_THRESHOLD,
    ANTIPODAL_THRESHOLD,
    DIVISION_EPS,
    ORTHOGONALITY_TOL,
    ORTHOGONALITY_TOL_FP16,
)
from .types import (
    RISEPrototypeResult,
    TransformPrediction,
    AlignmentScore,
    CrossLanguageTransferResult,
)

__all__ = [
    # Constants
    "ARCCOS_CLAMP_EPS",
    "NEAR_IDENTITY_THRESHOLD",
    "ANTIPODAL_THRESHOLD",
    "DIVISION_EPS",
    "ORTHOGONALITY_TOL",
    "ORTHOGONALITY_TOL_FP16",
    # Types
    "RISEPrototypeResult",
    "TransformPrediction",
    "AlignmentScore",
    "CrossLanguageTransferResult",
]
