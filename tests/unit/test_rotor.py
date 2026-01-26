"""
Unit tests for Householder rotor computation.

Tests the canonicalization operations used in RISE:
- Householder rotor computation
- Rotor application
- Orthogonality verification

Mathematical properties verified:
1. Rotor Orthogonality: R^T R = I
2. Rotor Mapping: R @ source = target
3. Rotor Determinant: det(R) = ±1
"""

import pytest
import torch
import torch.nn.functional as F

from rise.core.rotor import (
    compute_householder_rotor,
    apply_rotor,
    apply_rotor_transpose,
    verify_orthogonality,
    get_reference_direction,
)
from rise.utils.constants import (
    NEAR_IDENTITY_THRESHOLD,
    ROTOR_VERIFICATION_TOL,
    ROTOR_VERIFICATION_TOL_FP16,
)


class TestComputeHouseholderRotor:
    """Tests for computing Householder reflection matrices."""
    
    def test_rotor_maps_source_to_target(self, random_unit_vector):
        """Test that R @ source = target."""
        source = random_unit_vector(dim=512)
        target = random_unit_vector(dim=512)
        
        R = compute_householder_rotor(source, target, verify=True)
        result = apply_rotor(R, source)
        
        error = torch.norm(result - target)
        assert error < ROTOR_VERIFICATION_TOL, f"Rotor mapping error: {error}"
    
    def test_rotor_is_orthogonal(self, random_unit_vector):
        """Test that R^T R = I."""
        source = random_unit_vector(dim=512)
        target = random_unit_vector(dim=512)
        
        R = compute_householder_rotor(source, target)
        
        assert verify_orthogonality(R), "Rotor is not orthogonal"
    
    def test_rotor_to_default_target(self, random_unit_vector):
        """Test rotor mapping to default target e_1 = [1, 0, ..., 0]."""
        source = random_unit_vector(dim=512)
        
        R = compute_householder_rotor(source)
        result = apply_rotor(R, source)
        
        # Should map to e1
        e1 = get_reference_direction(512, device=source.device, dtype=source.dtype)
        error = torch.norm(result - e1)
        assert error < ROTOR_VERIFICATION_TOL
    
    def test_rotor_identity_case(self, random_unit_vector):
        """Test that R is identity when source = target."""
        source = random_unit_vector(dim=512)
        
        R = compute_householder_rotor(source, source)
        
        # Should be identity matrix
        identity = torch.eye(512, device=source.device, dtype=source.dtype)
        error = torch.norm(R - identity)
        assert error < NEAR_IDENTITY_THRESHOLD
    
    def test_rotor_antipodal_case(self):
        """Test rotor for antipodal vectors (source = -target)."""
        dim = 512
        source = torch.zeros(dim)
        source[0] = 1.0
        
        target = torch.zeros(dim)
        target[0] = -1.0
        
        R = compute_householder_rotor(source, target)
        result = apply_rotor(R, source)
        
        error = torch.norm(result - target)
        assert error < ROTOR_VERIFICATION_TOL
    
    def test_rotor_determinant(self, random_unit_vector):
        """Test that det(R) = ±1 (property of orthogonal matrices)."""
        source = random_unit_vector(dim=128)
        target = random_unit_vector(dim=128)
        
        R = compute_householder_rotor(source, target)
        det = torch.det(R)
        
        # Determinant should be ±1
        assert torch.abs(torch.abs(det) - 1.0) < 1e-5
    
    def test_rotor_different_dtypes(self, random_unit_vector):
        """Test rotor computation with float32 and float16."""
        for dtype in [torch.float32, torch.float16]:
            source = random_unit_vector(dim=128, dtype=dtype)
            target = random_unit_vector(dim=128, dtype=dtype)
            
            R = compute_householder_rotor(source, target)
            
            assert R.dtype == dtype
            assert verify_orthogonality(R)
    
    def test_rotor_high_dimensional(self):
        """Test rotor computation with OpenAI embedding dimensions."""
        torch.manual_seed(42)
        
        source = F.normalize(torch.randn(3072), dim=0)
        target = F.normalize(torch.randn(3072), dim=0)
        
        R = compute_householder_rotor(source, target, verify=True)
        
        assert R.shape == (3072, 3072)
        assert verify_orthogonality(R)
        
        result = apply_rotor(R, source)
        error = torch.norm(result - target)
        assert error < ROTOR_VERIFICATION_TOL


class TestApplyRotor:
    """Tests for rotor application operations."""
    
    def test_apply_rotor_preserves_norm(self, random_unit_vector):
        """Test that applying a rotor preserves vector norm."""
        source = random_unit_vector(dim=512)
        R = compute_householder_rotor(source)
        
        vector = torch.randn(512)
        original_norm = torch.norm(vector)
        
        result = apply_rotor(R, vector)
        result_norm = torch.norm(result)
        
        assert torch.allclose(original_norm, result_norm, atol=1e-5)
    
    def test_apply_rotor_transpose_is_inverse(self, random_unit_vector):
        """Test that R^T is the inverse of R."""
        source = random_unit_vector(dim=512)
        R = compute_householder_rotor(source)
        
        vector = torch.randn(512)
        
        # Apply R then R^T
        transformed = apply_rotor(R, vector)
        reconstructed = apply_rotor_transpose(R, transformed)
        
        error = torch.norm(reconstructed - vector)
        assert error < 1e-5
    
    def test_apply_rotor_batch(self, random_unit_vector_batch):
        """Test applying rotor to multiple vectors."""
        source = random_unit_vector_batch(1, dim=256)[0]
        R = compute_householder_rotor(source)
        
        vectors = torch.randn(10, 256)
        
        for vec in vectors:
            result = apply_rotor(R, vec)
            
            # Check norm preservation
            original_norm = torch.norm(vec)
            result_norm = torch.norm(result)
            assert torch.allclose(original_norm, result_norm, atol=1e-5)


class TestVerifyOrthogonality:
    """Tests for orthogonality verification."""
    
    def test_identity_is_orthogonal(self):
        """Test that identity matrix is orthogonal."""
        I = torch.eye(512)
        
        assert verify_orthogonality(I)
    
    def test_random_matrix_not_orthogonal(self):
        """Test that random matrix is not orthogonal."""
        R = torch.randn(512, 512)
        
        assert not verify_orthogonality(R)
    
    def test_rotation_matrix_is_orthogonal(self):
        """Test that a 2D rotation matrix is orthogonal."""
        import math
        
        theta = math.pi / 4
        R = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ])
        
        assert verify_orthogonality(R)
    
    def test_householder_rotor_is_orthogonal(self, random_unit_vector):
        """Test that Householder rotors are orthogonal."""
        source = random_unit_vector(dim=256)
        target = random_unit_vector(dim=256)
        
        R = compute_householder_rotor(source, target)
        
        assert verify_orthogonality(R)
    
    def test_orthogonality_with_custom_tolerance(self, random_unit_vector):
        """Test orthogonality verification with custom tolerance."""
        source = random_unit_vector(dim=128)
        R = compute_householder_rotor(source)
        
        # Should pass with loose tolerance
        assert verify_orthogonality(R, tol=1e-3)
        
        # Create slightly non-orthogonal matrix
        R_noisy = R + torch.randn_like(R) * 1e-4
        
        # Should fail with tight tolerance
        assert not verify_orthogonality(R_noisy, tol=1e-6)
    
    def test_orthogonality_different_dtypes(self):
        """Test orthogonality verification for float32 and float16."""
        for dtype in [torch.float32, torch.float16]:
            I = torch.eye(128, dtype=dtype)
            assert verify_orthogonality(I)


class TestGetReferenceDirection:
    """Tests for reference direction generation."""
    
    def test_reference_direction_is_e1(self):
        """Test that reference direction is [1, 0, ..., 0]."""
        e1 = get_reference_direction(512)
        
        expected = torch.zeros(512)
        expected[0] = 1.0
        
        assert torch.allclose(e1, expected)
    
    def test_reference_direction_is_unit(self):
        """Test that reference direction has unit norm."""
        e1 = get_reference_direction(512)
        
        norm = torch.norm(e1)
        assert torch.allclose(norm, torch.tensor(1.0))
    
    def test_reference_direction_different_dims(self):
        """Test reference direction with different dimensions."""
        for dim in [64, 512, 1024, 3072]:
            e1 = get_reference_direction(dim)
            
            assert e1.shape == (dim,)
            assert e1[0] == 1.0
            assert torch.allclose(e1[1:], torch.zeros(dim - 1))
    
    def test_reference_direction_dtype(self):
        """Test reference direction with different dtypes."""
        for dtype in [torch.float32, torch.float16, torch.float64]:
            e1 = get_reference_direction(512, dtype=dtype)
            
            assert e1.dtype == dtype


class TestRotorProperties:
    """Tests for mathematical properties of rotors."""
    
    def test_rotor_involution(self, random_unit_vector):
        """Test that applying a reflection twice is identity (R @ R = I)."""
        # For Householder reflections H = I - 2vv^T, we have H^2 = I
        source = random_unit_vector(dim=256)
        
        R = compute_householder_rotor(source)
        
        # R @ R should be identity (reflection applied twice)
        R_squared = R @ R
        identity = torch.eye(256, device=R.device, dtype=R.dtype)
        
        error = torch.norm(R_squared - identity)
        assert error < 1e-5
    
    def test_rotor_composition(self, random_unit_vector):
        """Test composing two rotors."""
        v1 = random_unit_vector(dim=256)
        v2 = random_unit_vector(dim=256)
        v3 = random_unit_vector(dim=256)
        
        # R1 maps v1 -> v2
        R1 = compute_householder_rotor(v1, v2)
        
        # R2 maps v2 -> v3
        R2 = compute_householder_rotor(v2, v3)
        
        # Composition R2 @ R1 should map v1 -> v3
        R_composed = R2 @ R1
        result = apply_rotor(R_composed, v1)
        
        error = torch.norm(result - v3)
        assert error < 1e-4
        
        # Composed rotor should still be orthogonal
        assert verify_orthogonality(R_composed, tol=1e-4)
    
    def test_rotor_preserves_dot_products(self, random_unit_vector):
        """Test that rotors preserve dot products (isometry property)."""
        source = random_unit_vector(dim=256)
        R = compute_householder_rotor(source)
        
        # Create two random vectors
        u = torch.randn(256)
        v = torch.randn(256)
        
        original_dot = torch.dot(u, v)
        
        # Apply rotor to both
        u_rotated = apply_rotor(R, u)
        v_rotated = apply_rotor(R, v)
        
        rotated_dot = torch.dot(u_rotated, v_rotated)
        
        assert torch.allclose(original_dot, rotated_dot, atol=1e-5)
    
    def test_rotor_preserves_angles(self, random_unit_vector_batch):
        """Test that rotors preserve angles between vectors."""
        source = random_unit_vector_batch(1, dim=256)[0]
        R = compute_householder_rotor(source)
        
        u = random_unit_vector_batch(1, dim=256)[0]
        v = random_unit_vector_batch(1, dim=256)[0]
        
        original_angle = torch.acos(torch.dot(u, v).clamp(-1, 1))
        
        u_rotated = apply_rotor(R, u)
        v_rotated = apply_rotor(R, v)
        u_rotated = F.normalize(u_rotated, dim=0)
        v_rotated = F.normalize(v_rotated, dim=0)
        
        rotated_angle = torch.acos(torch.dot(u_rotated, v_rotated).clamp(-1, 1))
        
        assert torch.allclose(original_angle, rotated_angle, atol=1e-5)


class TestRotorEdgeCases:
    """Tests for edge cases in rotor computation."""
    
    def test_rotor_with_zero_vector(self):
        """Test that rotor handles zero vector gracefully."""
        source = torch.zeros(512)
        
        # This should either handle it gracefully or raise an error
        # After normalization, it will be problematic
        with pytest.raises((ValueError, RuntimeError)):
            # Normalize will create NaN
            source_normalized = F.normalize(source, dim=0)
            R = compute_householder_rotor(source_normalized)
    
    def test_rotor_standard_basis_vectors(self):
        """Test rotor between standard basis vectors."""
        dim = 512
        
        e1 = torch.zeros(dim)
        e1[0] = 1.0
        
        e2 = torch.zeros(dim)
        e2[1] = 1.0
        
        R = compute_householder_rotor(e1, e2)
        result = apply_rotor(R, e1)
        
        error = torch.norm(result - e2)
        assert error < ROTOR_VERIFICATION_TOL
    
    def test_rotor_numerical_stability(self):
        """Test rotor computation with near-parallel vectors."""
        dim = 512
        
        source = F.normalize(torch.randn(dim), dim=0)
        
        # Create target very close to source
        epsilon = 1e-6
        target = source + torch.randn(dim) * epsilon
        target = F.normalize(target, dim=0)
        
        R = compute_householder_rotor(source, target)
        
        # Should be close to identity
        identity = torch.eye(dim, device=R.device, dtype=R.dtype)
        error = torch.norm(R - identity)
        
        # Should still be a valid rotor
        assert verify_orthogonality(R)
