"""
Unit tests for the high-level RISE interface.

Tests the RISE class that provides end-to-end functionality:
- Fitting with text pairs or embeddings
- Transforming new inputs
- Evaluation
- Save/load

This tests the integration of all RISE components.
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
import tempfile

from rise.core.rise import RISE
from rise.utils.types import RISEPrototypeResult, TransformPrediction, AlignmentScore


class TestRISEInit:
    """Tests for RISE initialization."""
    
    def test_init_without_embedder(self):
        """Test initializing RISE without an embedder."""
        rise = RISE()
        
        assert rise.embedder is None
        assert rise.canonicalize is True
        assert rise.is_fitted is False
    
    def test_init_with_embedder(self, mock_embedder):
        """Test initializing RISE with an embedder."""
        embedder = mock_embedder(dim=512)
        rise = RISE(embedder=embedder)
        
        assert rise.embedder is not None
        assert rise.is_fitted is False
    
    def test_init_with_canonicalize_flag(self):
        """Test initializing with canonicalize flag."""
        rise = RISE(canonicalize=False)
        
        assert rise.canonicalize is False


class TestRISEFit:
    """Tests for RISE fitting."""
    
    def test_fit_with_embeddings(self, sample_embedding_pairs):
        """Test fitting with pre-computed embeddings."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE()
        result = rise.fit(
            neutral_embeddings=neutral,
            transformed_embeddings=transformed
        )
        
        assert isinstance(result, RISEPrototypeResult)
        assert rise.is_fitted is True
        assert result.num_pairs == 10
    
    def test_fit_with_text_pairs(self, mock_batch_embedder):
        """Test fitting with text pairs."""
        embedder = mock_batch_embedder(dim=512)
        rise = RISE(embedder=embedder)
        
        training_pairs = [
            ("The sky is blue.", "The sky is not blue."),
            ("I like coffee.", "I don't like coffee."),
            ("It's sunny today.", "It's not sunny today."),
        ]
        
        result = rise.fit(training_pairs=training_pairs)
        
        assert isinstance(result, RISEPrototypeResult)
        assert rise.is_fitted is True
        assert result.num_pairs == 3
    
    def test_fit_without_embedder_raises_error(self):
        """Test that fitting with text pairs without embedder raises error."""
        rise = RISE()
        
        training_pairs = [
            ("Text 1", "Text 2"),
        ]
        
        with pytest.raises(ValueError, match="no embedder"):
            rise.fit(training_pairs=training_pairs)
    
    def test_fit_without_data_raises_error(self):
        """Test that fitting without data raises error."""
        rise = RISE()
        
        with pytest.raises(ValueError, match="Must provide"):
            rise.fit()
    
    def test_fit_updates_fitted_flag(self, sample_embedding_pairs):
        """Test that fitting updates the is_fitted flag."""
        neutral, transformed = sample_embedding_pairs(num_pairs=5, dim=256)
        
        rise = RISE()
        assert rise.is_fitted is False
        
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        assert rise.is_fitted is True
    
    def test_fit_returns_prototype_norm(self, sample_embedding_pairs):
        """Test that fit returns prototype with norm."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE()
        result = rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        assert result.prototype_norm > 0
        assert rise.prototype_norm > 0
        assert result.prototype_norm == rise.prototype_norm


class TestRISETransform:
    """Tests for RISE transformation."""
    
    def test_transform_with_embedding(self, sample_embedding_pairs, random_unit_vector):
        """Test transformation with pre-computed embedding."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        query = random_unit_vector(dim=512)
        prediction = rise.transform(embedding=query)
        
        assert isinstance(prediction, TransformPrediction)
        
        # Check that prediction is unit vector
        norm = torch.norm(prediction.predicted_embedding)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)
    
    def test_transform_with_text(self, sample_embedding_pairs, mock_embedder):
        """Test transformation with text input."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        embedder = mock_embedder(dim=512)
        rise = RISE(embedder=embedder)
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        prediction = rise.transform(text="A test sentence")
        
        assert isinstance(prediction, TransformPrediction)
    
    def test_transform_before_fit_raises_error(self, random_unit_vector):
        """Test that transforming before fitting raises error."""
        rise = RISE()
        
        query = random_unit_vector(dim=512)
        
        with pytest.raises(ValueError, match="not fitted"):
            rise.transform(embedding=query)
    
    def test_transform_without_embedder_raises_error(self, sample_embedding_pairs):
        """Test that transforming text without embedder raises error."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        with pytest.raises(ValueError, match="no embedder"):
            rise.transform(text="A test sentence")
    
    def test_transform_without_input_raises_error(self, sample_embedding_pairs):
        """Test that transforming without input raises error."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        with pytest.raises(ValueError, match="Must provide"):
            rise.transform()


class TestRISEEvaluate:
    """Tests for RISE evaluation."""
    
    def test_evaluate_with_embeddings(self, sample_embedding_pairs):
        """Test evaluation with pre-computed embeddings."""
        # Training data
        neutral_train, transformed_train = sample_embedding_pairs(num_pairs=20, dim=512)
        
        # Test data
        neutral_test, transformed_test = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral_train, transformed_embeddings=transformed_train)
        
        score = rise.evaluate(
            neutral_embeddings=neutral_test,
            transformed_embeddings=transformed_test
        )
        
        assert isinstance(score, AlignmentScore)
        assert -1.0 <= score.score <= 1.0
        assert score.std >= 0
        assert score.num_samples == 10
    
    def test_evaluate_with_text_pairs(self, sample_embedding_pairs, mock_batch_embedder):
        """Test evaluation with text pairs."""
        neutral_train, transformed_train = sample_embedding_pairs(num_pairs=10, dim=512)
        
        embedder = mock_batch_embedder(dim=512)
        rise = RISE(embedder=embedder)
        rise.fit(neutral_embeddings=neutral_train, transformed_embeddings=transformed_train)
        
        test_pairs = [
            ("Sentence 1", "Transformed 1"),
            ("Sentence 2", "Transformed 2"),
        ]
        
        score = rise.evaluate(test_pairs=test_pairs)
        
        assert isinstance(score, AlignmentScore)
        assert score.num_samples == 2
    
    def test_evaluate_before_fit_raises_error(self, sample_embedding_pairs):
        """Test that evaluating before fitting raises error."""
        neutral_test, transformed_test = sample_embedding_pairs(num_pairs=5, dim=512)
        
        rise = RISE()
        
        with pytest.raises(ValueError, match="not fitted"):
            rise.evaluate(
                neutral_embeddings=neutral_test,
                transformed_embeddings=transformed_test
            )
    
    def test_evaluate_return_per_sample(self, sample_embedding_pairs):
        """Test evaluation with per-sample scores."""
        neutral_train, transformed_train = sample_embedding_pairs(num_pairs=10, dim=512)
        neutral_test, transformed_test = sample_embedding_pairs(num_pairs=5, dim=512)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral_train, transformed_embeddings=transformed_train)
        
        score = rise.evaluate(
            neutral_embeddings=neutral_test,
            transformed_embeddings=transformed_test,
            return_per_sample=True
        )
        
        assert score.per_sample_scores is not None
        assert len(score.per_sample_scores) == 5
        
        # Each score should be in [-1, 1]
        for s in score.per_sample_scores:
            assert -1.0 <= s <= 1.0


class TestRISESaveLoad:
    """Tests for RISE persistence."""
    
    def test_save_and_load(self, sample_embedding_pairs):
        """Test save and load functionality."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        rise = RISE(canonicalize=True)
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rise_model.pt"
            rise.save(save_path)
            
            # Create new instance and load
            rise_loaded = RISE()
            rise_loaded.load(save_path)
        
        assert rise_loaded.is_fitted is True
        assert rise_loaded.prototype_norm == rise.prototype_norm
    
    def test_save_before_fit_raises_error(self):
        """Test that saving before fitting raises error."""
        rise = RISE()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rise_model.pt"
            
            with pytest.raises(ValueError, match="not fitted"):
                rise.save(save_path)
    
    def test_loaded_model_predictions_match(
        self,
        sample_embedding_pairs,
        random_unit_vector
    ):
        """Test that loaded model gives same predictions."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        query = random_unit_vector(dim=512)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        original_prediction = rise.transform(embedding=query)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rise_model.pt"
            rise.save(save_path)
            
            rise_loaded = RISE()
            rise_loaded.load(save_path)
        
        loaded_prediction = rise_loaded.transform(embedding=query)
        
        assert torch.allclose(
            original_prediction.predicted_embedding,
            loaded_prediction.predicted_embedding,
            atol=1e-6
        )


class TestRISEIntegration:
    """Integration tests for full RISE workflow."""
    
    def test_full_workflow_with_embeddings(self, sample_embedding_pairs, random_unit_vector):
        """Test complete workflow: fit, transform, evaluate."""
        # Generate data
        neutral_train, transformed_train = sample_embedding_pairs(num_pairs=20, dim=512)
        neutral_test, transformed_test = sample_embedding_pairs(num_pairs=10, dim=512)
        
        # Fit
        rise = RISE(canonicalize=True)
        fit_result = rise.fit(
            neutral_embeddings=neutral_train,
            transformed_embeddings=transformed_train
        )
        
        assert fit_result.num_pairs == 20
        
        # Transform
        query = random_unit_vector(dim=512)
        prediction = rise.transform(embedding=query)
        
        assert torch.norm(prediction.predicted_embedding).item() == pytest.approx(1.0, abs=1e-5)
        
        # Evaluate
        eval_result = rise.evaluate(
            neutral_embeddings=neutral_test,
            transformed_embeddings=transformed_test
        )
        
        assert eval_result.num_samples == 10
    
    def test_full_workflow_with_text(self, mock_batch_embedder):
        """Test complete workflow with text inputs."""
        embedder = mock_batch_embedder(dim=256)
        rise = RISE(embedder=embedder, canonicalize=True)
        
        # Training pairs
        training_pairs = [
            ("I like this", "I don't like this"),
            ("It works", "It doesn't work"),
            ("Good idea", "Bad idea"),
        ]
        
        rise.fit(training_pairs=training_pairs)
        
        # Transform
        prediction = rise.transform(text="This is great")
        assert isinstance(prediction, TransformPrediction)
        
        # Evaluate
        test_pairs = [
            ("Test 1", "Test 1 negated"),
            ("Test 2", "Test 2 negated"),
        ]
        
        eval_result = rise.evaluate(test_pairs=test_pairs)
        assert eval_result.num_samples == 2
    
    def test_workflow_with_and_without_canonicalization(self, sample_embedding_pairs):
        """Test that both canonicalization modes work."""
        neutral, transformed = sample_embedding_pairs(num_pairs=15, dim=512)
        query = F.normalize(torch.randn(512), dim=0)
        
        for canonicalize in [True, False]:
            rise = RISE(canonicalize=canonicalize)
            rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
            
            prediction = rise.transform(embedding=query)
            
            # Both should produce valid predictions
            assert torch.norm(prediction.predicted_embedding).item() == pytest.approx(1.0, abs=1e-5)
    
    def test_consistency_across_multiple_transforms(self, sample_embedding_pairs):
        """Test that multiple transforms of same input give same result."""
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=256)
        
        rise = RISE()
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        query = F.normalize(torch.randn(256), dim=0)
        
        # Transform multiple times
        pred1 = rise.transform(embedding=query)
        pred2 = rise.transform(embedding=query)
        pred3 = rise.transform(embedding=query)
        
        # Should all be identical
        assert torch.allclose(pred1.predicted_embedding, pred2.predicted_embedding)
        assert torch.allclose(pred2.predicted_embedding, pred3.predicted_embedding)


class TestRISEEmbedder:
    """Tests for embedder handling."""
    
    def test_single_sentence_embedder(self, mock_embedder):
        """Test with embedder that handles single sentences."""
        embedder = mock_embedder(dim=256)
        rise = RISE(embedder=embedder)
        
        # Should handle batch by calling single embedder multiple times
        training_pairs = [
            ("A", "B"),
            ("C", "D"),
        ]
        
        result = rise.fit(training_pairs=training_pairs)
        assert result.num_pairs == 2
    
    def test_batch_embedder(self, mock_batch_embedder):
        """Test with embedder that handles batches."""
        embedder = mock_batch_embedder(dim=256)
        rise = RISE(embedder=embedder)
        
        training_pairs = [
            ("A", "B"),
            ("C", "D"),
        ]
        
        result = rise.fit(training_pairs=training_pairs)
        assert result.num_pairs == 2
    
    def test_embedder_dimension_mismatch(self, mock_embedder, sample_embedding_pairs):
        """Test that dimension mismatch is caught."""
        # Fit with 512-dim embeddings
        neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
        
        # But embedder produces 256-dim
        embedder = mock_embedder(dim=256)
        rise = RISE(embedder=embedder)
        rise.fit(neutral_embeddings=neutral, transformed_embeddings=transformed)
        
        # Trying to transform with 256-dim will cause shape mismatch
        # The prediction will fail because dimensions don't match
        with pytest.raises((RuntimeError, ValueError)):
            rise.transform(text="test")
