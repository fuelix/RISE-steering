"""
Baseline steering methods for comparison with RISE.

This module contains implementations of comparison methods:
- Park's Linear Representation method (Park et al., 2024)
- CAA (Contrastive Activation Addition) (Rimsky et al., 2023)
- HPR (Householder Pseudo-Rotation) (Chai et al., 2024)
"""

from .base import SteeringMethod, SteeringResult
from .park import ParkMethod
from .caa import CAAMethod
from .hpr import HPRMethod

__all__ = [
    # Base classes
    "SteeringMethod",
    "SteeringResult",
    # Implementations
    "ParkMethod",
    "CAAMethod",
    "HPRMethod",
]
