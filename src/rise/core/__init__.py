"""
Core RISE implementation.

This module contains the fundamental components of RISE:
- Riemannian geometry operations on the hypersphere
- Householder rotor computation for canonicalization
- Prototype learning and transport
- High-level RISE interface
"""

from .rise import RISE
from .prototype import RISEPrototype, learn_rise_prototype, predict_transformation
from .riemannian import riemannian_log, riemannian_exp, geodesic_distance, project_to_tangent_space
from .rotor import compute_householder_rotor, apply_rotor, apply_rotor_transpose

__all__ = [
    "RISE",
    "RISEPrototype",
    "learn_rise_prototype",
    "predict_transformation",
    "riemannian_log",
    "riemannian_exp",
    "geodesic_distance",
    "project_to_tangent_space",
    "compute_householder_rotor",
    "apply_rotor",
    "apply_rotor_transpose",
]
