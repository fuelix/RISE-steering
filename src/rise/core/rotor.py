"""
Householder rotor computation for RISE canonicalization.

This module implements the canonicalization step of RISE, which uses Householder
reflections to align different sentence embeddings to a common reference frame.

Mathematical Background:
    A Householder reflection is an orthogonal transformation that reflects vectors
    across a hyperplane. Given a unit vector v, the Householder matrix is:

        H = I - 2 * v * v^T

    This matrix reflects any vector across the hyperplane orthogonal to v.

    In RISE, we use Householder reflections to map each neutral embedding n to
    a canonical reference direction e_1 = [1, 0, ..., 0]. This allows us to
    compare semantic transformations across different sentence contexts in a
    common coordinate frame.

References:
    - Householder, A.S. (1958). "Unitary Triangularization of a Nonsymmetric Matrix"
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from ..utils.constants import (
    NEAR_IDENTITY_THRESHOLD,
    ROTOR_VERIFICATION_TOL,
    ROTOR_VERIFICATION_TOL_FP16,
)

logger = logging.getLogger(__name__)


def compute_householder_rotor(
    source: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    verify: bool = False,
) -> torch.Tensor:
    """
    Compute a Householder reflection matrix that maps source to target.

    By default, target is e_1 = [1, 0, ..., 0], the first standard basis vector.
    The resulting orthogonal matrix R satisfies: R @ source = target.

    Formula:
        v = normalize(source - target)
        R = I - 2 * v * v^T

    Special cases:
        - If source ≈ target: returns identity matrix
        - If source ≈ -target: returns matrix that flips the first coordinate

    Args:
        source: Unit vector to be mapped, shape [d]
        target: Target unit vector (default: e_1), shape [d]
        verify: If True, verify that R @ source ≈ target

    Returns:
        Orthogonal matrix R such that R @ source = target, shape [d, d]
    """
    source = F.normalize(source, dim=0)
    d = len(source)
    device = source.device
    dtype = source.dtype

    # Default target is e_1
    if target is None:
        target = torch.zeros(d, device=device, dtype=dtype)
        target[0] = 1.0
    else:
        target = F.normalize(target, dim=0)

    # Check if source is already aligned with target
    if torch.allclose(source, target, atol=NEAR_IDENTITY_THRESHOLD):
        return torch.eye(d, device=device, dtype=dtype)

    # Check if source is opposite to target (special case)
    if torch.allclose(source, -target, atol=NEAR_IDENTITY_THRESHOLD):
        # Return a matrix that flips the sign of the first coordinate
        R = torch.eye(d, device=device, dtype=dtype)
        R[0, 0] = -1.0
        return R

    # Compute Householder reflection
    # v = (source - target) / ||source - target||
    v = source - target
    v = F.normalize(v, dim=0)

    # R = I - 2 * v * v^T
    R = torch.eye(d, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)

    # Verify the mapping if requested
    if verify:
        result = R @ source
        tol = ROTOR_VERIFICATION_TOL_FP16 if dtype == torch.float16 else ROTOR_VERIFICATION_TOL
        error = torch.norm(result - target)
        if error >= tol:
            logger.warning(
                f"Rotor verification: ||R @ source - target|| = {error:.2e} (tol={tol:.0e})"
            )

    return R


def apply_rotor(
    rotor: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a rotor (orthogonal matrix) to a vector.

    This is simply matrix-vector multiplication, but provided as a named function
    for clarity in the RISE algorithm.

    Args:
        rotor: Orthogonal matrix R, shape [d, d]
        vector: Vector to transform, shape [d]

    Returns:
        Transformed vector R @ vector, shape [d]
    """
    return rotor @ vector


def apply_rotor_transpose(
    rotor: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the transpose of a rotor to a vector.

    For orthogonal matrices, R^T = R^{-1}, so this is the inverse transformation.
    Used in RISE to transport the prototype back from the canonical frame.

    Args:
        rotor: Orthogonal matrix R, shape [d, d]
        vector: Vector to transform, shape [d]

    Returns:
        Transformed vector R^T @ vector, shape [d]
    """
    return rotor.T @ vector


def verify_orthogonality(
    matrix: torch.Tensor,
    tol: Optional[float] = None,
) -> bool:
    """
    Verify that a matrix is orthogonal (R^T R = I).

    Args:
        matrix: Matrix to verify, shape [d, d]
        tol: Tolerance for comparison (default based on dtype)

    Returns:
        True if matrix is orthogonal within tolerance
    """
    if tol is None:
        tol = ROTOR_VERIFICATION_TOL_FP16 if matrix.dtype == torch.float16 else ROTOR_VERIFICATION_TOL

    d = matrix.shape[0]
    identity = torch.eye(d, device=matrix.device, dtype=matrix.dtype)
    product = matrix.T @ matrix

    # Use max-element error (infinity norm) for dimension-independent check
    error = torch.max(torch.abs(product - identity))
    return error.item() < tol


def get_reference_direction(
    dim: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Get the canonical reference direction e_1 = [1, 0, ..., 0].

    This is the target direction used in RISE canonicalization.

    Args:
        dim: Dimensionality of the embedding space
        device: Torch device for the tensor
        dtype: Torch dtype for the tensor

    Returns:
        Unit vector e_1, shape [dim]
    """
    e1 = torch.zeros(dim, device=device, dtype=dtype)
    e1[0] = 1.0
    return e1
