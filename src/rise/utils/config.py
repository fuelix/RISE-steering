"""
Configuration management for RISE experiments.

This module provides a hierarchical configuration system using dataclasses
for type safety and YAML files for persistence.
"""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""

    model_name: str = "text-embedding-3-large"
    """Name of the embedding model to use."""

    dimension: int = 3072
    """Embedding dimensionality."""

    normalize: bool = True
    """Whether to L2-normalize embeddings."""

    batch_size: int = 32
    """Batch size for embedding extraction."""

    cache_embeddings: bool = True
    """Whether to cache embeddings to disk."""

    cache_dir: str = "data/processed/embeddings"
    """Directory for cached embeddings."""


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    languages: List[str] = field(default_factory=lambda: [
        "en", "es", "ja", "ta", "th", "ar", "zu"
    ])
    """Languages to include in experiments."""

    transformations: List[str] = field(default_factory=lambda: [
        "negation", "conditionality", "politeness"
    ])
    """Transformation types to evaluate."""

    train_samples_per_lang: int = 800
    """Number of training samples per language."""

    test_samples_per_lang: int = 200
    """Number of test samples per language."""

    min_sentence_length: int = 5
    """Minimum sentence length in tokens."""

    max_sentence_length: int = 25
    """Maximum sentence length in tokens."""

    data_dir: str = "data/raw"
    """Directory containing raw data files."""

    random_seed: int = 42
    """Random seed for data splitting."""


@dataclass
class RISEConfig:
    """Configuration for RISE method."""

    canonicalize: bool = True
    """Whether to use canonicalization."""

    verify_tangent_space: bool = True
    """Whether to verify tangent space constraints."""

    dtype: str = "float32"
    """Data type for computations (float32 or float16)."""


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    metrics: List[str] = field(default_factory=lambda: [
        "alignment_score", "cross_language_transfer", "centroid_similarity"
    ])
    """Metrics to compute."""

    num_folds: int = 5
    """Number of cross-validation folds."""

    compute_std: bool = True
    """Whether to compute standard deviations."""

    save_per_sample: bool = False
    """Whether to save per-sample scores."""


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment."""

    name: str = "rise_experiment"
    """Experiment name."""

    description: str = ""
    """Experiment description."""

    output_dir: str = "outputs"
    """Directory for experiment outputs."""

    random_seed: int = 42
    """Global random seed for reproducibility."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    """Embedding configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    """Data configuration."""

    rise: RISEConfig = field(default_factory=RISEConfig)
    """RISE method configuration."""

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    """Evaluation configuration."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use JSON for now (YAML requires pyyaml dependency)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML/JSON file."""
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        return cls(
            name=data.get("name", "rise_experiment"),
            description=data.get("description", ""),
            output_dir=data.get("output_dir", "outputs"),
            random_seed=data.get("random_seed", 42),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            data=DataConfig(**data.get("data", {})),
            rise=RISEConfig(**data.get("rise", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )


def get_default_config() -> ExperimentConfig:
    """Get the default experiment configuration."""
    return ExperimentConfig()


def get_iclr_config() -> ExperimentConfig:
    """Get the configuration used for ICLR 2026 submission."""
    return ExperimentConfig(
        name="iclr_2026_submission",
        description="Geometric Rotor Interpretations of Multilingual Embedding Models",
        embedding=EmbeddingConfig(
            model_name="text-embedding-3-large",
            dimension=3072,
        ),
        data=DataConfig(
            languages=["en", "es", "ja", "ta", "th", "ar", "zu"],
            transformations=["negation", "conditionality", "politeness"],
            train_samples_per_lang=800,
            test_samples_per_lang=200,
        ),
        rise=RISEConfig(
            canonicalize=True,
            dtype="float32",
        ),
        evaluation=EvaluationConfig(
            metrics=["alignment_score", "cross_language_transfer", "centroid_similarity"],
            num_folds=5,
        ),
    )
