"""
Unit tests for Riemannian geometry operations.

Tests the fundamental operations on the unit hypersphere S^{d-1}:
- Logarithmic map (riemannian_log)
- Exponential map (riemannian_exp)
- Geodesic distance
- Tangent space projection

Mathematical properties verified:
1. Log-Exp inverse: exp_n(log_n(v)) = v
2. Exp-Log inverse: log_n(exp_n(ξ)) = ξ when ξ ⊥ n
3. Distance = Norm: geodesic_distance(n, v) = ||log_n(v)||
4. Tangent Orthogonality: log_n(v) ⊥ n
"""

import pytest
import torch
import torch.nn.functional as F
import math

from rise.core.riemannian import (
    riemannian_log,
    riemannian_exp,
    geodesic_distance,
    project_to_tangent_space,
)
from rise.utils.constants import (
    NEAR_IDENTITY_THRESHOLD,
    ANTIPODAL_THRESHOLD,
    ORTHOGONALITY_TOL,
    ORTHOGONALITY_TOL_FP16,
)


class TestRiemannianLog:
    """Tests for the logarithmic map."""
    
    def test_log_returns_tangent_vector(self, random_unit_vector):
        """Test that log map returns a vector orthogonal to the base."""
        base = random_unit_vector(dim=512)
        target = random_unit_vector(dim=512)
        
        log_result = riemannian_log(base, target, verify_tangent_space=True)
        
        # Verify orthogonality: <log_result, base> ≈ 0
        dot_product = torch.abs(torch.dot(log_result, base))
        assert dot_product < ORTHOGONALITY_TOL, f"Log result not orthogonal: {dot_product}"
    
    def test_log_norm_equals_geodesic_distance(self, random_unit_vector):
        """Test that ||log_n(v)|| = geodesic_distance(n, v)."""
        base = random_unit_vector(dim=512)
        target = random_unit_vector(dim=512)
        
        log_result = riemannian_log(base, target)
        log_norm = torch.norm(log_result)
        
        geo_dist = geodesic_distance(base, target)
        
        # These should be equal
        assert torch.allclose(log_norm, geo_dist, atol=1e-6), \
            f"Log norm {log_norm} != geodesic distance {geo_dist}"
    
    def test_log_near_identity_returns_zero(self, near_vectors):
        """Test that log of near-identical vectors returns near-zero vector."""
        base, target = near_vectors(dim=512, angle_degrees=0.001)
        
        log_result = riemannian_log(base, target)
        
        # Should be very small
        assert torch.norm(log_result) < NEAR_IDENTITY_THRESHOLD * 10
    
    def test_log_antipodal_raises_error(self, antipodal_vectors):
        """Test that log of antipodal vectors raises ValueError."""
        base, target = antipodal_vectors(dim=512)
        
        with pytest.raises(ValueError, match="nearly antipodal"):
            riemannian_log(base, target)
    
    def test_log_different_dtypes(self, random_unit_vector):
        """Test log map with float32 and float16."""
        for dtype in [torch.float32, torch.float16]:
            base = random_unit_vector(dim=128, dtype=dtype)
            target = random_unit_vector(dim=128, dtype=dtype)
            
            log_result = riemannian_log(base, target, verify_tangent_space=True)
            
            assert log_result.dtype == dtype
            
            # Check orthogonality with appropriate tolerance
            tol = ORTHOGONALITY_TOL_FP16 if dtype == torch.float16 else ORTHOGONALITY_TOL
            dot_product = torch.abs(torch.dot(log_result, base))
            assert dot_product < tol
    
    def test_log_high_dimensional(self, random_unit_vector):
        """Test log map works with high-dimensional vectors (like OpenAI embeddings)."""
        base = random_unit_vector(dim=3072)
        target = random_unit_vector(dim=3072)
        
        log_result = riemannian_log(base, target)
        
        # Basic sanity checks
        assert log_result.shape == (3072,)
        assert not torch.isnan(log_result).any()
        assert not torch.isinf(log_result).any()


class TestRiemannianExp:
    """Tests for the exponential map."""
    
    def test_exp_returns_unit_vector(self, random_unit_vector):
        """Test that exp map returns a point on the unit sphere."""
        base = random_unit_vector(dim=512)
        
        # Generate a random tangent vector
        tangent = torch.randn(512)
        tangent = tangent - torch.dot(tangent, base) * base  # Make orthogonal
        tangent = tangent * 0.5  # Scale to reasonable magnitude
        
        exp_result = riemannian_exp(base, tangent, verify_tangent_space=True)
        
        # Should be unit norm
        norm = torch.norm(exp_result)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6), \
            f"Exp result not on unit sphere: norm = {norm}"
    
    def test_exp_zero_tangent_returns_base(self, random_unit_vector):
        """Test that exp with zero tangent vector returns the base point."""
        base = random_unit_vector(dim=512)
        tangent = torch.zeros(512)
        
        exp_result = riemannian_exp(base, tangent)
        
        assert torch.allclose(exp_result, base, atol=1e-6)
    
    def test_exp_invalid_tangent_raises_error(self, random_unit_vector):
        """Test that exp with non-tangent vector raises ValueError."""
        base = random_unit_vector(dim=512)
        
        # Use base itself as tangent (not orthogonal)
        invalid_tangent = base.clone()
        
        with pytest.raises(ValueError, match="not in tangent space"):
            riemannian_exp(base, invalid_tangent, verify_tangent_space=True)
    
    def test_exp_different_dtypes(self, random_unit_vector):
        """Test exp map with float32 and float16."""
        for dtype in [torch.float32, torch.float16]:
            base = random_unit_vector(dim=128, dtype=dtype)
            
            tangent = torch.randn(128, dtype=dtype)
            tangent = tangent - torch.dot(tangent, base) * base
            tangent = tangent * 0.3
            
            exp_result = riemannian_exp(base, tangent, verify_tangent_space=True)
            
            assert exp_result.dtype == dtype
            
            # Check unit norm
            norm = torch.norm(exp_result)
            assert torch.allclose(norm, torch.tensor(1.0, dtype=dtype), atol=1e-3)


class TestLogExpInverse:
    """Tests for the fundamental inverse relationships."""
    
    def test_exp_log_inverse(self, random_unit_vector):
        """Test that exp_n(log_n(v)) = v."""
        base = random_unit_vector(dim=512)
        target = random_unit_vector(dim=512)
        
        # Compute log then exp
        log_result = riemannian_log(base, target)
        reconstructed = riemannian_exp(base, log_result)
        
        # Should recover target
        assert torch.allclose(reconstructed, target, atol=1e-5), \
            f"Exp-Log inverse failed: error = {torch.norm(reconstructed - target)}"
    
    def test_log_exp_inverse(self, random_unit_vector):
        """Test that log_n(exp_n(ξ)) = ξ when ξ ⊥ n."""
        base = random_unit_vector(dim=512)
        
        # Generate tangent vector
        tangent = torch.randn(512)
        tangent = tangent - torch.dot(tangent, base) * base  # Make orthogonal
        tangent = tangent * 0.5
        
        # Compute exp then log
        point = riemannian_exp(base, tangent)
        reconstructed = riemannian_log(base, point)
        
        # Should recover tangent
        assert torch.allclose(reconstructed, tangent, atol=1e-5), \
            f"Log-Exp inverse failed: error = {torch.norm(reconstructed - tangent)}"
    
    def test_round_trip_multiple_points(self, random_unit_vector_batch):
        """Test round-trip consistency for multiple points."""
        base = random_unit_vector_batch(1, dim=256)[0]
        targets = random_unit_vector_batch(20, dim=256)
        
        for target in targets:
            log_result = riemannian_log(base, target)
            reconstructed = riemannian_exp(base, log_result)
            
            error = torch.norm(reconstructed - target)
            assert error < 1e-5, f"Round-trip error: {error}"
    
    def test_round_trip_high_dimensional(self):
        """Test round-trip with OpenAI embedding dimensions."""
        torch.manual_seed(123)
        
        base = F.normalize(torch.randn(3072), dim=0)
        target = F.normalize(torch.randn(3072), dim=0)
        
        log_result = riemannian_log(base, target)
        reconstructed = riemannian_exp(base, log_result)
        
        error = torch.norm(reconstructed - target)
        assert error < 1e-5


class TestGeodesicDistance:
    """Tests for geodesic distance computation."""
    
    def test_distance_is_symmetric(self, random_unit_vector):
        """Test that d(a, b) = d(b, a)."""
        a = random_unit_vector(dim=512)
        b = random_unit_vector(dim=512)
        
        dist_ab = geodesic_distance(a, b)
        dist_ba = geodesic_distance(b, a)
        
        assert torch.allclose(dist_ab, dist_ba, atol=1e-6)
    
    def test_distance_to_self_is_zero(self, random_unit_vector):
        """Test that d(a, a) = 0."""
        a = random_unit_vector(dim=512)
        
        dist = geodesic_distance(a, a)
        
        assert dist < NEAR_IDENTITY_THRESHOLD
    
    def test_distance_in_valid_range(self, random_unit_vector):
        """Test that distance is in [0, π]."""
        a = random_unit_vector(dim=512)
        b = random_unit_vector(dim=512)
        
        dist = geodesic_distance(a, b)
        
        assert 0 <= dist <= math.pi
    
    def test_distance_antipodal_is_pi(self, antipodal_vectors):
        """Test that distance between antipodal points is π."""
        a, b = antipodal_vectors(dim=512)
        
        dist = geodesic_distance(a, b)
        
        assert torch.allclose(dist, torch.tensor(math.pi), atol=1e-5)
    
    def test_distance_matches_arccos_formula(self, random_unit_vector):
        """Test that distance matches arccos(a · b)."""
        a = random_unit_vector(dim=512)
        b = random_unit_vector(dim=512)
        
        dist = geodesic_distance(a, b)
        
        # Manual computation
        cos_dist = torch.dot(a, b).clamp(-1.0, 1.0)
        expected_dist = torch.acos(cos_dist)
        
        assert torch.allclose(dist, expected_dist, atol=1e-6)


class TestProjectToTangentSpace:
    """Tests for tangent space projection."""
    
    def test_projection_is_orthogonal(self, random_unit_vector):
        """Test that projected vector is orthogonal to base."""
        base = random_unit_vector(dim=512)
        vector = torch.randn(512)
        
        projected = project_to_tangent_space(vector, base)
        
        dot_product = torch.abs(torch.dot(projected, base))
        assert dot_product < ORTHOGONALITY_TOL
    
    def test_projection_idempotent(self, random_unit_vector):
        """Test that projecting twice gives same result."""
        base = random_unit_vector(dim=512)
        vector = torch.randn(512)
        
        projected_once = project_to_tangent_space(vector, base)
        projected_twice = project_to_tangent_space(projected_once, base)
        
        assert torch.allclose(projected_once, projected_twice, atol=1e-6)
    
    def test_projection_of_orthogonal_unchanged(self, random_unit_vector):
        """Test that projecting already-orthogonal vector doesn't change it."""
        base = random_unit_vector(dim=512)
        
        # Create orthogonal vector
        vector = torch.randn(512)
        vector = vector - torch.dot(vector, base) * base
        
        projected = project_to_tangent_space(vector, base)
        
        assert torch.allclose(projected, vector, atol=1e-6)
    
    def test_projection_removes_parallel_component(self, random_unit_vector):
        """Test that projection removes the component parallel to base."""
        base = random_unit_vector(dim=512)
        
        # Vector with known parallel component
        vector = base * 2.5 + torch.randn(512) * 0.1
        
        projected = project_to_tangent_space(vector, base)
        
        # Parallel component should be gone
        dot_product = torch.abs(torch.dot(projected, base))
        assert dot_product < ORTHOGONALITY_TOL


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_small_angles(self, near_vectors):
        """Test operations with very small angular separations."""
        base, target = near_vectors(dim=512, angle_degrees=0.01)
        
        log_result = riemannian_log(base, target)
        reconstructed = riemannian_exp(base, log_result)
        
        error = torch.norm(reconstructed - target)
        assert error < 1e-5
    
    def test_right_angle_separation(self):
        """Test with vectors at 90 degrees (π/2)."""
        # Create orthogonal vectors
        base = torch.zeros(512)
        base[0] = 1.0
        
        target = torch.zeros(512)
        target[1] = 1.0
        
        dist = geodesic_distance(base, target)
        assert torch.allclose(dist, torch.tensor(math.pi / 2), atol=1e-5)
        
        log_result = riemannian_log(base, target)
        assert torch.allclose(torch.norm(log_result), dist, atol=1e-5)
    
    def test_numerical_stability_near_boundary(self):
        """Test numerical stability near the identity threshold."""
        base = F.normalize(torch.randn(512), dim=0)
        
        # Create target very close to base
        epsilon = NEAR_IDENTITY_THRESHOLD * 0.5
        tangent = torch.randn(512)
        tangent = tangent - torch.dot(tangent, base) * base
        tangent = F.normalize(tangent, dim=0) * epsilon
        
        target = base + tangent
        target = F.normalize(target, dim=0)
        
        # Should not crash
        log_result = riemannian_log(base, target)
        assert not torch.isnan(log_result).any()
