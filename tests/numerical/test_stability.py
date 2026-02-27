"""
Numerical stability and property-based tests for RISE.

Uses hypothesis for property-based testing to verify RISE operations
are numerically stable across a wide range of inputs:
- Random high-dimensional vectors
- Different dtypes (float32, float16)
- Boundary conditions
- Extreme values

This provides confidence that RISE works reliably in production.
"""

import pytest
import torch
import torch.nn.functional as F
import math
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import numpy as np

from rise.core.riemannian import (
    riemannian_log,
    riemannian_exp,
    geodesic_distance,
    project_to_tangent_space,
)
from rise.core.rotor import (
    compute_householder_rotor,
    apply_rotor,
    verify_orthogonality,
)
from rise.core.prototype import RISEPrototype
from rise.utils.constants import (
    NEAR_IDENTITY_THRESHOLD,
    ORTHOGONALITY_TOL,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

@st.composite
def unit_vector(draw, dim=None):
    """Generate a random unit vector."""
    if dim is None:
        dim = draw(st.integers(min_value=2, max_value=1024))
    
    vec = draw(arrays(
        dtype=np.float32,
        shape=(dim,),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ))
    
    # Convert to torch and normalize
    vec_tensor = torch.from_numpy(vec)
    norm = torch.norm(vec_tensor)
    assume(norm > 1e-6)  # Avoid zero vectors
    
    return F.normalize(vec_tensor, dim=0)


@st.composite
def unit_vector_pair(draw, dim=None):
    """Generate a pair of unit vectors that aren't too close to antipodal."""
    if dim is None:
        dim = draw(st.integers(min_value=2, max_value=512))
    
    v1 = draw(unit_vector(dim=dim))
    v2 = draw(unit_vector(dim=dim))
    
    # Avoid nearly antipodal or nearly identical pairs (float32 edge cases)
    cos_angle = torch.dot(v1, v2)
    assume(cos_angle > -0.99)  # Not too close to antipodal
    assume(cos_angle < 0.99)   # Not too close to identical

    return v1, v2


# =============================================================================
# Property-Based Tests: Riemannian Operations
# =============================================================================

class TestRiemannianProperties:
    """Property-based tests for Riemannian operations."""
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_log_exp_inverse_property(self, vectors):
        """Property: exp_n(log_n(v)) = v for all unit vectors n, v."""
        base, target = vectors
        
        log_result = riemannian_log(base, target)
        reconstructed = riemannian_exp(base, log_result)
        
        error = torch.norm(reconstructed - target)
        assert error < 1e-4, f"Log-Exp inverse failed with error {error}"
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_log_result_orthogonal_property(self, vectors):
        """Property: log_n(v) ⊥ n for all unit vectors n, v."""
        base, target = vectors
        
        log_result = riemannian_log(base, target, verify_tangent_space=False)
        
        dot_product = torch.abs(torch.dot(log_result, base))
        assert dot_product < ORTHOGONALITY_TOL, \
            f"Log result not orthogonal: <ξ, n> = {dot_product}"
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_distance_equals_log_norm_property(self, vectors):
        """Property: geodesic_distance(n, v) = ||log_n(v)|| for all n, v."""
        base, target = vectors
        
        log_result = riemannian_log(base, target)
        log_norm = torch.norm(log_result)
        
        geo_dist = geodesic_distance(base, target)
        
        assert torch.allclose(log_norm, geo_dist, atol=1e-4), \
            f"Distance {geo_dist} != log norm {log_norm}"
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_distance_symmetric_property(self, vectors):
        """Property: d(a, b) = d(b, a) for all a, b."""
        a, b = vectors
        
        dist_ab = geodesic_distance(a, b)
        dist_ba = geodesic_distance(b, a)
        
        assert torch.allclose(dist_ab, dist_ba, atol=1e-6)
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_exp_returns_unit_vector_property(self, vectors):
        """Property: exp_n(ξ) has unit norm for all n, ξ."""
        base, _ = vectors
        
        # Generate random tangent vector
        tangent = torch.randn_like(base)
        tangent = tangent - torch.dot(tangent, base) * base
        tangent = tangent * 0.5  # Scale to reasonable magnitude
        
        exp_result = riemannian_exp(base, tangent)
        norm = torch.norm(exp_result)
        
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5), \
            f"Exp result not unit: norm = {norm}"
    
    @given(unit_vector())
    @settings(deadline=None, max_examples=50)
    def test_projection_orthogonal_property(self, base):
        """Property: project(v, n) ⊥ n for all v, n."""
        vector = torch.randn_like(base)
        
        projected = project_to_tangent_space(vector, base)
        
        dot_product = torch.abs(torch.dot(projected, base))
        assert dot_product < ORTHOGONALITY_TOL


# =============================================================================
# Property-Based Tests: Rotor Operations
# =============================================================================

class TestRotorProperties:
    """Property-based tests for rotor operations."""
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_rotor_orthogonality_property(self, vectors):
        """Property: R^T R = I for all rotors R."""
        source, target = vectors
        
        R = compute_householder_rotor(source, target)
        
        assert verify_orthogonality(R), "Rotor not orthogonal"
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_rotor_mapping_property(self, vectors):
        """Property: R @ source = target for all source, target."""
        source, target = vectors
        
        R = compute_householder_rotor(source, target)
        result = apply_rotor(R, source)
        
        error = torch.norm(result - target)
        assert error < 1e-3, f"Rotor mapping failed with error {error}"
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_rotor_preserves_norm_property(self, vectors):
        """Property: ||R @ v|| = ||v|| for all rotors R and vectors v."""
        source, target = vectors
        
        R = compute_householder_rotor(source, target)
        
        # Random vector
        v = torch.randn_like(source)
        original_norm = torch.norm(v)
        
        transformed = apply_rotor(R, v)
        transformed_norm = torch.norm(transformed)
        
        assert torch.allclose(original_norm, transformed_norm, atol=1e-5)
    
    @given(unit_vector_pair())
    @settings(deadline=None, max_examples=50)
    def test_rotor_involution_property(self, vectors):
        """Property: R @ R = I for Householder reflections."""
        source, _ = vectors
        
        R = compute_householder_rotor(source)
        
        # R squared should be identity
        R_squared = R @ R
        identity = torch.eye(len(source), device=R.device, dtype=R.dtype)

        error = torch.max(torch.abs(R_squared - identity))
        assert error < 1e-4


# =============================================================================
# Property-Based Tests: RISE Prototype
# =============================================================================

class TestRISEPrototypeProperties:
    """Property-based tests for RISE prototype learning."""
    
    @given(
        st.integers(min_value=5, max_value=20),
        st.integers(min_value=64, max_value=512),
    )
    @settings(deadline=None, max_examples=20)
    def test_prototype_prediction_unit_norm_property(self, num_pairs, dim):
        """Property: RISE predictions always have unit norm."""
        torch.manual_seed(42)
        
        # Generate training data
        neutral = F.normalize(torch.randn(num_pairs, dim), dim=1)
        transformed = F.normalize(torch.randn(num_pairs, dim), dim=1)
        
        # Learn and predict
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        # Test on multiple queries
        for _ in range(5):
            query = F.normalize(torch.randn(dim), dim=0)
            prediction = prototype.predict(query)
            
            norm = torch.norm(prediction.predicted_embedding)
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-4)
    
    @given(
        st.integers(min_value=5, max_value=20),
        st.integers(min_value=64, max_value=256),
    )
    @settings(deadline=None, max_examples=20)
    def test_prototype_determinism_property(self, num_pairs, dim):
        """Property: Learning is deterministic with same data."""
        torch.manual_seed(42)
        
        neutral = F.normalize(torch.randn(num_pairs, dim), dim=1)
        transformed = F.normalize(torch.randn(num_pairs, dim), dim=1)
        
        # Learn twice
        proto1 = RISEPrototype()
        result1 = proto1.learn(neutral, transformed)
        
        proto2 = RISEPrototype()
        result2 = proto2.learn(neutral, transformed)
        
        assert torch.allclose(result1.prototype, result2.prototype, atol=1e-6)


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_high_dimensional_vectors(self):
        """Test with very high-dimensional vectors (e.g., 8192)."""
        torch.manual_seed(42)
        
        dim = 8192
        base = F.normalize(torch.randn(dim), dim=0)
        target = F.normalize(torch.randn(dim), dim=0)
        
        # Log-exp roundtrip
        log_result = riemannian_log(base, target)
        reconstructed = riemannian_exp(base, log_result)
        
        error = torch.norm(reconstructed - target)
        assert error < 1e-4
    
    def test_very_small_dimensions(self):
        """Test with small dimensions (2D, 3D)."""
        for dim in [2, 3]:
            base = F.normalize(torch.randn(dim), dim=0)
            target = F.normalize(torch.randn(dim), dim=0)
            
            log_result = riemannian_log(base, target)
            reconstructed = riemannian_exp(base, log_result)
            
            error = torch.norm(reconstructed - target)
            assert error < 1e-5
    
    def test_near_zero_tangent_vectors(self):
        """Test with very small tangent vectors."""
        base = F.normalize(torch.randn(512), dim=0)
        
        # Very small tangent
        tangent = torch.randn(512) * 1e-8
        tangent = tangent - torch.dot(tangent, base) * base
        
        exp_result = riemannian_exp(base, tangent)
        
        # Should be very close to base
        error = torch.norm(exp_result - base)
        assert error < 1e-6
    
    def test_large_geodesic_distances(self):
        """Test with large geodesic separations (close to π)."""
        torch.manual_seed(123)

        base = F.normalize(torch.randn(512), dim=0)

        # Create target that is exactly antipodal — log map is undefined here
        target = -base

        with pytest.raises(ValueError, match="[Aa]ntipodal"):
            riemannian_log(base, target)
    
    def test_numerical_precision_float16(self):
        """Test numerical precision with float16."""
        base = F.normalize(torch.randn(256, dtype=torch.float16), dim=0)
        target = F.normalize(torch.randn(256, dtype=torch.float16), dim=0)
        
        log_result = riemannian_log(base, target)
        reconstructed = riemannian_exp(base, log_result)
        
        # Looser tolerance for float16
        error = torch.norm(reconstructed - target)
        assert error < 1e-2
    
    def test_batch_operations_stability(self):
        """Test stability with batch operations."""
        torch.manual_seed(42)
        
        dim = 512
        batch_size = 100
        
        neutral = F.normalize(torch.randn(batch_size, dim), dim=1)
        transformed = F.normalize(torch.randn(batch_size, dim), dim=1)
        
        # Learn prototype
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        # Verify prototype is valid
        assert not torch.isnan(result.prototype).any()
        assert not torch.isinf(result.prototype).any()
        assert result.prototype_norm > 0


# =============================================================================
# Stress Tests
# =============================================================================

class TestStressConditions:
    """Stress tests for RISE under challenging conditions."""
    
    def test_many_training_pairs(self):
        """Test with many training pairs."""
        torch.manual_seed(42)
        
        dim = 256
        num_pairs = 1000
        
        neutral = F.normalize(torch.randn(num_pairs, dim), dim=1)
        transformed = F.normalize(torch.randn(num_pairs, dim), dim=1)
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        assert result.num_pairs == num_pairs
        assert not torch.isnan(result.prototype).any()
    
    def test_repeated_predictions(self):
        """Test making many predictions."""
        torch.manual_seed(42)
        
        dim = 256
        neutral = F.normalize(torch.randn(10, dim), dim=1)
        transformed = F.normalize(torch.randn(10, dim), dim=1)
        
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        # Make many predictions
        for _ in range(100):
            query = F.normalize(torch.randn(dim), dim=0)
            prediction = prototype.predict(query)
            
            # All should be valid
            assert torch.norm(prediction.predicted_embedding).item() == pytest.approx(1.0, abs=1e-4)
    
    def test_identical_training_pairs(self):
        """Test with identical neutral and transformed embeddings."""
        torch.manual_seed(42)
        
        dim = 256
        neutral = F.normalize(torch.randn(10, dim), dim=1)
        transformed = neutral.clone()  # Identical
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        # Prototype should be near-zero (no transformation)
        assert result.prototype_norm < 0.1
    
    def test_random_transformations_consistency(self):
        """Test that random transformations still produce consistent results."""
        torch.manual_seed(42)
        
        dim = 512
        num_pairs = 50
        
        neutral = F.normalize(torch.randn(num_pairs, dim), dim=1)
        
        # Random transformations
        transformed = []
        for i in range(num_pairs):
            tangent = torch.randn(dim)
            tangent = tangent - torch.dot(tangent, neutral[i]) * neutral[i]
            tangent = F.normalize(tangent, dim=0) * 0.5
            
            v = math.cos(0.5) * neutral[i] + math.sin(0.5) * tangent / 0.5
            transformed.append(F.normalize(v, dim=0))
        
        transformed = torch.stack(transformed)
        
        # Should learn successfully
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        # Predictions should be valid
        query = F.normalize(torch.randn(dim), dim=0)
        prediction = prototype.predict(query)
        
        assert torch.norm(prediction.predicted_embedding).item() == pytest.approx(1.0, abs=1e-4)
