"""
Unit tests for RISE prototype learning and prediction.

Tests the core RISE algorithm:
- Prototype learning from embedding pairs
- Prediction on new embeddings
- Save/load functionality
- Canonicalization vs non-canonicalization

Mathematical properties verified:
1. Prototype Unit Norm: Predictions are on unit sphere
2. Learning Convergence: Valid prototypes are learned
3. Persistence: Save/load preserves prototype state
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
import tempfile

from rise.core.prototype import (
    RISEPrototype,
    learn_rise_prototype,
    predict_transformation,
)
from rise.utils.types import RISEPrototypeResult, TransformPrediction


class TestRISEPrototypeLearn:
    """Tests for prototype learning."""
    
    def test_learn_basic(self, sample_embedding_pairs):
        """Test basic prototype learning."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed, canonicalize=True)
        
        assert isinstance(result, RISEPrototypeResult)
        assert result.prototype is not None
        assert result.num_pairs == 10
        assert result.canonicalized is True
        assert result.prototype.shape == (512,)
    
    def test_learn_without_canonicalization(self, sample_embedding_pairs):
        """Test learning without canonicalization."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed, canonicalize=False)
        
        assert result.canonicalized is False
        assert result.prototype is not None
    
    def test_learn_mismatched_shapes_raises_error(self, random_unit_vector_batch):
        """Test that mismatched shapes raise ValueError."""
        neutral = random_unit_vector_batch(10, dim=512)
        transformed = random_unit_vector_batch(5, dim=512)
        
        prototype = RISEPrototype()
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            prototype.learn(neutral, transformed)
    
    def test_learn_stores_metadata(self, sample_embedding_pairs):
        """Test that learning stores correct metadata."""
        neutral, transformed = sample_embedding_pairs(num_pairs=20, dim=256)
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        assert prototype.dim == 256
        assert prototype._num_training_pairs == 20
        assert prototype.canonicalized is True
    
    def test_learn_with_high_dimensional_embeddings(self, sample_embedding_pairs):
        """Test learning with OpenAI embedding dimensions."""
        neutral, transformed = sample_embedding_pairs(num_pairs=15, dim=3072)
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        assert result.prototype.shape == (3072,)
        assert not torch.isnan(result.prototype).any()
        assert not torch.isinf(result.prototype).any()
    
    def test_learn_prototype_norm(self, sample_embedding_pairs):
        """Test that learned prototype has reasonable norm."""
        neutral, transformed = sample_embedding_pairs(
            num_pairs=10, 
            dim=512,
            angular_shift=0.5
        )
        
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        # Prototype norm should be positive and reasonable
        assert result.prototype_norm > 0
        assert result.prototype_norm < 5.0  # Sanity check
    
    def test_learn_different_dtypes(self, sample_embedding_pairs):
        """Test learning with float32 and float16."""
        for dtype in [torch.float32, torch.float16]:
            neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=256, dtype=dtype)
            
            prototype = RISEPrototype()
            result = prototype.learn(neutral, transformed)
            
            assert result.prototype.dtype == dtype


class TestRISEPrototypePredict:
    """Tests for prototype prediction."""
    
    def test_predict_returns_unit_vector(self, sample_embedding_pairs, random_unit_vector):
        """Test that predictions are unit vectors."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        # Predict on new embedding
        new_neutral = random_unit_vector(dim=512)
        prediction = prototype.predict(new_neutral)
        
        assert isinstance(prediction, TransformPrediction)
        
        # Check unit norm
        norm = torch.norm(prediction.predicted_embedding)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)
    
    def test_predict_before_learn_raises_error(self, random_unit_vector):
        """Test that predicting before learning raises ValueError."""
        prototype = RISEPrototype()
        
        neutral = random_unit_vector(dim=512)
        
        with pytest.raises(ValueError, match="not learned"):
            prototype.predict(neutral)
    
    def test_predict_includes_source(self, sample_embedding_pairs, random_unit_vector):
        """Test that prediction includes source embedding."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        new_neutral = random_unit_vector(dim=512)
        prediction = prototype.predict(new_neutral)
        
        assert torch.allclose(prediction.source_embedding, new_neutral, atol=1e-6)
    
    def test_predict_cosine_similarity(self, sample_embedding_pairs, random_unit_vector):
        """Test that prediction includes cosine similarity."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        new_neutral = random_unit_vector(dim=512)
        prediction = prototype.predict(new_neutral)
        
        # Cosine similarity should be in [-1, 1]
        assert -1.0 <= prediction.cosine_similarity <= 1.0
        
        # Verify it matches manual computation
        manual_cos_sim = F.cosine_similarity(
            new_neutral,
            prediction.predicted_embedding,
            dim=0
        ).item()
        
        assert abs(prediction.cosine_similarity - manual_cos_sim) < 1e-5
    
    def test_predict_without_canonicalization(self, sample_embedding_pairs, random_unit_vector):
        """Test prediction without canonicalization."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed, canonicalize=False)
        
        new_neutral = random_unit_vector(dim=512)
        prediction = prototype.predict(new_neutral)
        
        # Should still return unit vector
        norm = torch.norm(prediction.predicted_embedding)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)
    
    def test_predict_multiple_queries(self, sample_embedding_pairs, random_unit_vector_batch):
        """Test prediction on multiple queries."""
        neutral, transformed = sample_embedding_pairs(num_pairs=15, dim=256)
        
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        queries = random_unit_vector_batch(20, dim=256)
        
        for query in queries:
            prediction = prototype.predict(query)
            
            # All predictions should be unit vectors
            norm = torch.norm(prediction.predicted_embedding)
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)


class TestRISEPrototypeSaveLoad:
    """Tests for prototype persistence."""
    
    def test_save_and_load_roundtrip(self, sample_embedding_pairs, random_unit_vector):
        """Test that save/load preserves prototype."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        # Learn and save
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "prototype.pt"
            prototype.save(save_path)
            
            # Load
            loaded_prototype = RISEPrototype.load(save_path)
        
        # Verify prototype is preserved
        assert torch.allclose(loaded_prototype.prototype, prototype.prototype)
        assert loaded_prototype.canonicalized == prototype.canonicalized
        assert loaded_prototype.dim == prototype.dim
        assert loaded_prototype._num_training_pairs == prototype._num_training_pairs
    
    def test_save_before_learn_raises_error(self):
        """Test that saving before learning raises ValueError."""
        prototype = RISEPrototype()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "prototype.pt"
            
            with pytest.raises(ValueError, match="No prototype to save"):
                prototype.save(save_path)
    
    def test_loaded_prototype_predictions_match(
        self,
        sample_embedding_pairs,
        random_unit_vector
    ):
        """Test that loaded prototype gives same predictions."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        # Learn original
        prototype = RISEPrototype()
        prototype.learn(neutral, transformed)
        
        # Make prediction
        query = random_unit_vector(dim=512)
        original_prediction = prototype.predict(query)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "prototype.pt"
            prototype.save(save_path)
            loaded_prototype = RISEPrototype.load(save_path)
        
        # Make prediction with loaded prototype
        loaded_prediction = loaded_prototype.predict(query)
        
        # Predictions should match
        assert torch.allclose(
            original_prediction.predicted_embedding,
            loaded_prediction.predicted_embedding,
            atol=1e-6
        )
    
    def test_save_preserves_canonicalization_flag(self, sample_embedding_pairs):
        """Test that save/load preserves canonicalization flag."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=256)
        
        for canonicalize in [True, False]:
            prototype = RISEPrototype()
            prototype.learn(neutral, transformed, canonicalize=canonicalize)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / f"prototype_canon_{canonicalize}.pt"
                prototype.save(save_path)
                loaded = RISEPrototype.load(save_path)
            
            assert loaded.canonicalized == canonicalize


class TestFunctionalAPI:
    """Tests for functional convenience functions."""
    
    def test_learn_rise_prototype_function(self, sample_embedding_pairs):
        """Test the learn_rise_prototype convenience function."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype_tensor = learn_rise_prototype(neutral, transformed, canonicalize=True)
        
        assert isinstance(prototype_tensor, torch.Tensor)
        assert prototype_tensor.shape == (512,)
    
    def test_predict_transformation_function(
        self,
        sample_embedding_pairs,
        random_unit_vector
    ):
        """Test the predict_transformation convenience function."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        prototype_tensor = learn_rise_prototype(neutral, transformed)
        
        query = random_unit_vector(dim=512)
        prediction = predict_transformation(prototype_tensor, query, canonicalized=True)
        
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == (512,)
        
        # Should be unit vector
        norm = torch.norm(prediction)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)
    
    def test_functional_api_matches_class_api(
        self,
        sample_embedding_pairs,
        random_unit_vector
    ):
        """Test that functional API gives same results as class API."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        query = random_unit_vector(dim=512)
        
        # Class API
        prototype_obj = RISEPrototype()
        prototype_obj.learn(neutral, transformed)
        class_prediction = prototype_obj.predict(query).predicted_embedding
        
        # Functional API
        prototype_tensor = learn_rise_prototype(neutral, transformed)
        func_prediction = predict_transformation(prototype_tensor, query)
        
        # Should match
        assert torch.allclose(class_prediction, func_prediction, atol=1e-5)


class TestPrototypeConsistency:
    """Tests for prototype consistency and stability."""
    
    def test_prototype_deterministic(self, sample_embedding_pairs):
        """Test that learning is deterministic with same data."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=256)
        
        # Learn twice
        proto1 = RISEPrototype()
        result1 = proto1.learn(neutral, transformed)
        
        proto2 = RISEPrototype()
        result2 = proto2.learn(neutral, transformed)
        
        # Should be identical
        assert torch.allclose(result1.prototype, result2.prototype, atol=1e-6)
    
    def test_prototype_scales_with_transformation_magnitude(self):
        """Test that prototype norm increases with larger transformations."""
        torch.manual_seed(42)
        
        dim = 256
        neutral = F.normalize(torch.randn(10, dim), dim=1)
        
        # Small transformations
        small_shift = 0.1
        transformed_small = []
        for i in range(10):
            tangent = torch.randn(dim)
            tangent = tangent - torch.dot(tangent, neutral[i]) * neutral[i]
            tangent = F.normalize(tangent, dim=0) * small_shift
            v = torch.cos(small_shift) * neutral[i] + torch.sin(small_shift) * tangent / small_shift
            transformed_small.append(F.normalize(v, dim=0))
        transformed_small = torch.stack(transformed_small)
        
        # Large transformations
        large_shift = 1.0
        transformed_large = []
        for i in range(10):
            tangent = torch.randn(dim)
            tangent = tangent - torch.dot(tangent, neutral[i]) * neutral[i]
            tangent = F.normalize(tangent, dim=0) * large_shift
            v = torch.cos(large_shift) * neutral[i] + torch.sin(large_shift) * tangent / large_shift
            transformed_large.append(F.normalize(v, dim=0))
        transformed_large = torch.stack(transformed_large)
        
        proto_small = RISEPrototype()
        result_small = proto_small.learn(neutral, transformed_small)
        
        proto_large = RISEPrototype()
        result_large = proto_large.learn(neutral, transformed_large)
        
        # Larger transformation should have larger prototype norm
        assert result_large.prototype_norm > result_small.prototype_norm
    
    def test_prototype_with_consistent_direction(self):
        """Test that prototype learns a consistent direction."""
        torch.manual_seed(123)
        
        dim = 256
        num_pairs = 20
        
        # Create synthetic data with consistent transformation direction
        neutral = F.normalize(torch.randn(num_pairs, dim), dim=1)
        
        # All transformations in the same direction (towards e1)
        e1 = torch.zeros(dim)
        e1[0] = 1.0
        
        transformed = []
        for i in range(num_pairs):
            # Move each neutral 0.3 radians towards e1
            n = neutral[i]
            
            # Tangent pointing towards e1
            tangent = e1 - torch.dot(e1, n) * n
            tangent = F.normalize(tangent, dim=0) * 0.3
            
            v = torch.cos(0.3) * n + torch.sin(0.3) * tangent / 0.3
            transformed.append(F.normalize(v, dim=0))
        
        transformed = torch.stack(transformed)
        
        # Learn prototype
        prototype = RISEPrototype()
        result = prototype.learn(neutral, transformed)
        
        # Prototype should have reasonable norm
        assert result.prototype_norm > 0.1
        assert result.prototype_norm < 1.0
