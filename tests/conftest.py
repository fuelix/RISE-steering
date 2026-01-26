"""
Pytest fixtures for RISE test suite.

This module provides reusable fixtures for generating test data:
- Random unit vectors on the hypersphere
- Sample embedding pairs for training/testing
- Mock embedder functions
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Callable, List


# =============================================================================
# Random Seed Configuration
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    yield
    # Reset after test completes
    torch.manual_seed(torch.initial_seed())


# =============================================================================
# Unit Vector Fixtures
# =============================================================================

@pytest.fixture
def random_unit_vector():
    """Factory fixture to generate random unit vectors."""
    def _generate(dim: int = 512, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Generate a random unit vector on S^{d-1}.
        
        Args:
            dim: Dimensionality of the vector
            dtype: Data type for the tensor
            
        Returns:
            Random unit vector of shape [dim]
        """
        vec = torch.randn(dim, dtype=dtype)
        return F.normalize(vec, dim=0)
    
    return _generate


@pytest.fixture
def random_unit_vector_batch():
    """Factory fixture to generate batches of random unit vectors."""
    def _generate(
        batch_size: int = 10,
        dim: int = 512,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generate a batch of random unit vectors on S^{d-1}.
        
        Args:
            batch_size: Number of vectors to generate
            dim: Dimensionality of each vector
            dtype: Data type for the tensor
            
        Returns:
            Batch of unit vectors of shape [batch_size, dim]
        """
        vecs = torch.randn(batch_size, dim, dtype=dtype)
        return F.normalize(vecs, dim=1)
    
    return _generate


@pytest.fixture
def near_vectors():
    """Factory fixture to generate pairs of nearly identical vectors."""
    def _generate(
        dim: int = 512,
        angle_degrees: float = 1.0,
        dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two unit vectors separated by a small angle.
        
        Args:
            dim: Dimensionality
            angle_degrees: Angle separation in degrees
            dtype: Data type
            
        Returns:
            Tuple of (base, target) vectors with small angular separation
        """
        import math
        
        # Start with a random base vector
        base = torch.randn(dim, dtype=dtype)
        base = F.normalize(base, dim=0)
        
        # Create a small perturbation
        angle_rad = math.radians(angle_degrees)
        tangent = torch.randn(dim, dtype=dtype)
        # Make tangent orthogonal to base
        tangent = tangent - torch.dot(tangent, base) * base
        tangent = F.normalize(tangent, dim=0) * angle_rad
        
        # Target is approximately base + tangent (then normalized)
        target = base + tangent
        target = F.normalize(target, dim=0)
        
        return base, target
    
    return _generate


@pytest.fixture
def antipodal_vectors():
    """Factory fixture to generate nearly antipodal vector pairs."""
    def _generate(
        dim: int = 512,
        dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two vectors that are nearly opposite on the sphere.
        
        Args:
            dim: Dimensionality
            dtype: Data type
            
        Returns:
            Tuple of (base, target) where target ≈ -base
        """
        base = torch.randn(dim, dtype=dtype)
        base = F.normalize(base, dim=0)
        target = -base
        
        return base, target
    
    return _generate


# =============================================================================
# Embedding Pair Fixtures
# =============================================================================

@pytest.fixture
def sample_embedding_pairs(random_unit_vector_batch):
    """Factory fixture to generate sample (neutral, transformed) embedding pairs."""
    def _generate(
        num_pairs: int = 10,
        dim: int = 512,
        angular_shift: float = 0.5,
        dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic embedding pairs for testing.
        
        Args:
            num_pairs: Number of pairs to generate
            dim: Embedding dimensionality
            angular_shift: Approximate geodesic distance of the shift (in radians)
            dtype: Data type
            
        Returns:
            Tuple of (neutral_embeddings, transformed_embeddings)
            each of shape [num_pairs, dim]
        """
        import math
        
        neutral = random_unit_vector_batch(num_pairs, dim, dtype)
        
        # Generate transformed embeddings by moving along a random tangent direction
        transformed = []
        for i in range(num_pairs):
            n = neutral[i]
            
            # Random tangent direction
            tangent = torch.randn(dim, dtype=dtype)
            tangent = tangent - torch.dot(tangent, n) * n  # Make orthogonal to n
            tangent = F.normalize(tangent, dim=0) * angular_shift
            
            # Move along geodesic
            t_norm = torch.norm(tangent)
            if t_norm > 1e-8:
                tangent_unit = tangent / t_norm
                v = torch.cos(t_norm) * n + torch.sin(t_norm) * tangent_unit
                v = F.normalize(v, dim=0)
            else:
                v = n
                
            transformed.append(v)
        
        transformed = torch.stack(transformed)
        
        return neutral, transformed
    
    return _generate


# =============================================================================
# Mock Embedder Fixtures
# =============================================================================

@pytest.fixture
def mock_embedder():
    """Factory fixture to create mock embedding functions."""
    def _create(dim: int = 512, dtype: torch.dtype = torch.float32) -> Callable:
        """
        Create a mock embedder that returns deterministic embeddings.
        
        Args:
            dim: Embedding dimensionality
            dtype: Data type
            
        Returns:
            Mock embedder function
        """
        def embedder(text: str) -> torch.Tensor:
            """Mock embedder that hashes text to a deterministic vector."""
            # Use hash of text as seed for reproducibility
            seed = hash(text) % (2**31)
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            vec = torch.randn(dim, dtype=dtype, generator=generator)
            return F.normalize(vec, dim=0)
        
        return embedder
    
    return _create


@pytest.fixture
def mock_batch_embedder():
    """Factory fixture to create mock batch embedding functions."""
    def _create(dim: int = 512, dtype: torch.dtype = torch.float32) -> Callable:
        """
        Create a mock batch embedder that returns deterministic embeddings.
        
        Args:
            dim: Embedding dimensionality
            dtype: Data type
            
        Returns:
            Mock batch embedder function
        """
        def batch_embedder(texts: List[str]) -> torch.Tensor:
            """Mock batch embedder that hashes texts to deterministic vectors."""
            embeddings = []
            for text in texts:
                seed = hash(text) % (2**31)
                generator = torch.Generator()
                generator.manual_seed(seed)
                
                vec = torch.randn(dim, dtype=dtype, generator=generator)
                vec = F.normalize(vec, dim=0)
                embeddings.append(vec)
            
            return torch.stack(embeddings)
        
        return batch_embedder
    
    return _create


# =============================================================================
# Standard Test Dimensions
# =============================================================================

@pytest.fixture(params=[64, 512, 3072])
def test_dimension(request):
    """Parametrized fixture for testing different embedding dimensions."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float16])
def test_dtype(request):
    """Parametrized fixture for testing different data types."""
    return request.param
