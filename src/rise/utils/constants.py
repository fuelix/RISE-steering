"""
Numerical constants and tolerances for RISE.

These constants define the numerical precision thresholds used throughout
the RISE implementation for Riemannian geometry operations.
"""

import math

# =============================================================================
# Numerical Tolerances
# =============================================================================

# Clamping epsilon for arccos to avoid NaN from numerical errors
# Must be small enough that acos(1-eps) ≈ 0 and acos(-1+eps) ≈ π to within
# useful precision. 1e-7 was too large (acos(1-1e-7) ≈ 4.5e-4).
ARCCOS_CLAMP_EPS = 0.0

# Threshold below which vectors are considered identical (theta ≈ 0)
NEAR_IDENTITY_THRESHOLD = 1e-6

# Threshold above which vectors are considered antipodal (theta ≈ π)
ANTIPODAL_THRESHOLD = math.pi - 1e-6

# Division safety epsilon for operations like 1/sin(theta)
DIVISION_EPS = 1e-8

# Tolerance for verifying tangent space orthogonality (float32)
# Must accommodate accumulated float32 error: O(sqrt(d) * eps_mach) ≈ 3e-6
# for d=512, but pathological inputs can be 10-50x worse.
ORTHOGONALITY_TOL = 1e-4

# Relaxed tolerance for float16 precision
ORTHOGONALITY_TOL_FP16 = 1e-2

# Tolerance for verifying rotor maps n to e1 and orthogonality (max-element)
ROTOR_VERIFICATION_TOL = 5e-5
ROTOR_VERIFICATION_TOL_FP16 = 1e-2

# =============================================================================
# Geometry Constants
# =============================================================================

# Reference direction for canonicalization (first basis vector)
# In code, we construct e1 dynamically to match tensor dimensions

# =============================================================================
# Default Model Configuration
# =============================================================================

DEFAULT_EMBEDDING_DIM = 3072  # OpenAI text-embedding-3-large
DEFAULT_DTYPE_STR = "float32"
