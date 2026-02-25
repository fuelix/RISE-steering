"""
Baseline methods for comparison with RISE.

MDV: Mean Difference Vector — averages Euclidean difference vectors.
Procrustes: Orthogonal Procrustes alignment via SVD.
"""

from .mdv import MDV
from .procrustes import Procrustes

__all__ = ["MDV", "Procrustes"]
