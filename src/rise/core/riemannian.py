"""
Riemannian geometry operations on the unit hypersphere S^{d-1}.

This module implements the fundamental Riemannian operations needed for RISE:
- Logarithmic map: computes tangent vectors between points on the sphere
- Exponential map: moves along geodesics from a base point
- Geodesic distance: arc length between points

Mathematical Background:
    Sentence embeddings (when L2-normalized) reside on the unit hypersphere S^{d-1}.
    The geodesic (shortest path) between two points on the sphere is an arc of a
    great circle. The logarithmic map computes the initial velocity vector needed
    to travel along this geodesic, and the exponential map traces the geodesic
    given an initial velocity.

References:
    - Do Carmo, M.P. (1992). Riemannian Geometry.
    - Lee, J.M. (2018). Introduction to Riemannian Manifolds.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from ..utils.constants import (
    ARCCOS_CLAMP_EPS,
    NEAR_IDENTITY_THRESHOLD,
    ANTIPODAL_THRESHOLD,
    DIVISION_EPS,
    ORTHOGONALITY_TOL,
    ORTHOGONALITY_TOL_FP16,
)

logger = logging.getLogger(__name__)


def riemannian_log(
    base: torch.Tensor,
    target: torch.Tensor,
    verify_tangent_space: bool = True,
) -> torch.Tensor:
    """
    Compute the Riemannian logarithmic map from base to target on S^{d-1}.

    Given two unit vectors n (base) and v (target) on the unit hypersphere,
    computes the tangent vector ξ at n that points toward v along the geodesic.

    Formula:
        log_n(v) = (θ / sin(θ)) * (v - cos(θ) * n)
        where θ = arccos(n · v) is the geodesic angle.

    The result ξ lies in the tangent space T_n S^{d-1}, which means ξ ⊥ n.
    The magnitude ||ξ|| = θ equals the geodesic distance.

    Args:
        base: Unit vector n ∈ S^{d-1}, shape [d]
        target: Unit vector v ∈ S^{d-1}, shape [d]
        verify_tangent_space: If True, verify the result is orthogonal to base

    Returns:
        Tangent vector ξ ∈ T_n S^{d-1}, shape [d]

    Raises:
        ValueError: If vectors are antipodal (θ ≈ π), as log map is undefined.

    Note:
        Both inputs should be L2-normalized. This function normalizes them
        internally for safety, but pre-normalized inputs are preferred.
    """
    # Ensure unit vectors
    base = F.normalize(base, dim=0)
    target = F.normalize(target, dim=0)

    # Compute geodesic angle
    cos_theta = torch.dot(base, target).clamp(-1 + ARCCOS_CLAMP_EPS, 1 - ARCCOS_CLAMP_EPS)
    theta = torch.acos(cos_theta)

    # Handle near-identity case (v ≈ n)
    if theta < NEAR_IDENTITY_THRESHOLD:
        return torch.zeros_like(base)

    # Handle antipodal case (v ≈ -n)
    if theta > ANTIPODAL_THRESHOLD:
        raise ValueError(
            f"Vectors are nearly antipodal (θ={theta:.6f} rad ≈ π). "
            "The logarithmic map is undefined for antipodal points."
        )

    # Compute tangent vector
    # The component of v orthogonal to n is: v - (n·v)n = v - cos(θ)n
    # This has magnitude sin(θ), and we scale to get magnitude θ
    sin_theta = torch.sin(theta)
    tangent_component = target - cos_theta * base
    log_result = (theta / sin_theta) * tangent_component

    # Verify tangent space constraint: ξ ⊥ n
    if verify_tangent_space:
        tol = ORTHOGONALITY_TOL_FP16 if base.dtype == torch.float16 else ORTHOGONALITY_TOL
        dot_product = torch.abs(torch.dot(log_result, base))
        if dot_product >= tol:
            logger.warning(
                f"Tangent space verification: <ξ, n> = {dot_product:.2e} (tol={tol:.0e})"
            )

    return log_result


def riemannian_exp(
    base: torch.Tensor,
    tangent: torch.Tensor,
    verify_tangent_space: bool = True,
) -> torch.Tensor:
    """
    Compute the Riemannian exponential map from base in direction tangent.

    Given a unit vector n (base) on the unit hypersphere and a tangent vector
    ξ ∈ T_n S^{d-1}, computes the point reached by traveling along the geodesic
    starting at n with initial velocity ξ.

    Formula:
        exp_n(ξ) = cos(||ξ||) * n + sin(||ξ||) * (ξ / ||ξ||)

    The geodesic distance traveled is ||ξ||.

    Args:
        base: Unit vector n ∈ S^{d-1}, shape [d]
        tangent: Tangent vector ξ ∈ T_n S^{d-1}, shape [d]
        verify_tangent_space: If True, verify tangent is orthogonal to base

    Returns:
        Unit vector on S^{d-1} at geodesic distance ||ξ|| from base, shape [d]

    Note:
        The tangent vector should be orthogonal to base (in the tangent space).
        This function verifies this constraint if verify_tangent_space=True.
    """
    # Ensure unit base point
    base = F.normalize(base, dim=0)

    # Handle zero tangent vector (stay at base)
    tangent_norm = torch.norm(tangent)
    if tangent_norm < DIVISION_EPS:
        return base

    # Verify tangent space constraint: ξ ⊥ n
    if verify_tangent_space:
        tol = ORTHOGONALITY_TOL_FP16 if base.dtype == torch.float16 else ORTHOGONALITY_TOL
        dot_product = torch.abs(torch.dot(tangent, base))
        if dot_product >= tol:
            raise ValueError(
                f"Tangent vector not in tangent space: <ξ, n> = {dot_product:.2e} (tol={tol:.0e})"
            )

    # Compute exponential map
    tangent_unit = tangent / tangent_norm
    result = torch.cos(tangent_norm) * base + torch.sin(tangent_norm) * tangent_unit

    # Ensure result is exactly on unit sphere (correct for numerical drift)
    result = F.normalize(result, dim=0)

    return result


def geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the geodesic (arc-length) distance between two points on S^{d-1}.

    The geodesic distance is the length of the shortest path (great circle arc)
    between two points on the sphere, which equals the angle between them.

    Formula:
        d(a, b) = arccos(a · b)

    Args:
        a: Unit vector on S^{d-1}, shape [d]
        b: Unit vector on S^{d-1}, shape [d]

    Returns:
        Geodesic distance in radians, scalar tensor in [0, π]
    """
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)

    cos_dist = torch.dot(a, b).clamp(-1 + ARCCOS_CLAMP_EPS, 1 - ARCCOS_CLAMP_EPS)
    return torch.acos(cos_dist)


def project_to_tangent_space(
    vector: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """
    Project an ambient vector onto the tangent space at base.

    The tangent space T_n S^{d-1} at point n consists of all vectors orthogonal
    to n. This function removes the component of vector parallel to base.

    Formula:
        proj_{T_n}(v) = v - (v · n) * n

    Args:
        vector: Vector in ambient R^d space, shape [d]
        base: Unit vector defining the tangent space, shape [d]

    Returns:
        Projected vector in T_base S^{d-1}, shape [d]
    """
    base = F.normalize(base, dim=0)
    parallel_component = torch.dot(vector, base)
    return vector - parallel_component * base
