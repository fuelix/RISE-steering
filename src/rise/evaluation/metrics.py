"""
Evaluation metrics for RISE experiments.

This module implements the metrics used in the ICLR paper:
- Alignment score (cosine similarity between predicted and actual)
- Cross-language transfer performance
- Centroid similarity across languages
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..utils.types import AlignmentScore, CrossLanguageTransferResult

logger = logging.getLogger(__name__)


def compute_alignment_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    return_per_sample: bool = False,
) -> AlignmentScore:
    """
    Compute alignment score between predictions and ground truth.

    The alignment score is the mean cosine similarity between predicted
    transformed embeddings and actual transformed embeddings.

    Args:
        predictions: Predicted embeddings, shape [N, d]
        targets: Ground truth embeddings, shape [N, d]
        return_per_sample: Whether to return individual scores.

    Returns:
        AlignmentScore with mean, std, and optionally per-sample scores.
    """
    # Normalize
    predictions = F.normalize(predictions, dim=1)
    targets = F.normalize(targets, dim=1)

    # Compute cosine similarity per sample
    scores = F.cosine_similarity(predictions, targets, dim=1)

    mean_score = scores.mean().item()
    std_score = scores.std().item()

    return AlignmentScore(
        score=mean_score,
        std=std_score,
        num_samples=len(scores),
        per_sample_scores=scores.tolist() if return_per_sample else None,
    )


def compute_cross_language_transfer(
    source_prototype: torch.Tensor,
    source_language: str,
    target_neutral: torch.Tensor,
    target_transformed: torch.Tensor,
    target_language: str,
    transformation_type: str,
    predict_fn,
) -> CrossLanguageTransferResult:
    """
    Evaluate cross-language transfer of a prototype.

    Tests whether a prototype learned on one language can successfully
    predict transformations in another language.

    Args:
        source_prototype: Prototype learned on source language.
        source_language: Language code of source.
        target_neutral: Neutral embeddings in target language, [N, d].
        target_transformed: Transformed embeddings in target language, [N, d].
        target_language: Language code of target.
        transformation_type: Type of transformation.
        predict_fn: Function to predict transformation given prototype and embedding.

    Returns:
        CrossLanguageTransferResult with transfer score.
    """
    predictions = []
    for i in range(len(target_neutral)):
        pred = predict_fn(source_prototype, target_neutral[i])
        predictions.append(pred)

    predictions = torch.stack(predictions)
    alignment = compute_alignment_score(predictions, target_transformed)

    return CrossLanguageTransferResult(
        source_language=source_language,
        target_language=target_language,
        alignment_score=alignment.score,
        transformation_type=transformation_type,
    )


def compute_centroid_similarity(
    centroids: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise cosine similarity between language centroids.

    The centroid is the average canonicalized transformation vector
    for a language. High similarity indicates that the transformation
    occupies similar geometric directions across languages.

    Args:
        centroids: Dictionary mapping language codes to centroid vectors.

    Returns:
        Dictionary mapping language pairs to similarity scores.
    """
    languages = list(centroids.keys())
    similarities = {}

    for i, lang1 in enumerate(languages):
        for lang2 in languages[i:]:
            c1 = F.normalize(centroids[lang1], dim=0)
            c2 = F.normalize(centroids[lang2], dim=0)
            sim = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
            similarities[(lang1, lang2)] = sim

    return similarities


@dataclass
class TransformationResults:
    """Results for a single transformation type across languages."""

    transformation: str
    """Transformation type (negation, conditionality, politeness)."""

    per_language_scores: Dict[str, AlignmentScore]
    """Alignment scores per language."""

    cross_language_matrix: Dict[Tuple[str, str], float]
    """Cross-language transfer scores."""

    centroid_similarities: Dict[Tuple[str, str], float]
    """Centroid similarity scores."""

    mean_score: float
    """Mean alignment score across languages."""

    std_score: float
    """Standard deviation of scores across languages."""


def aggregate_results(
    per_language: Dict[str, AlignmentScore],
) -> Tuple[float, float]:
    """
    Aggregate per-language results into mean and std.

    Args:
        per_language: Dictionary of alignment scores per language.

    Returns:
        (mean, std) across languages.
    """
    scores = [s.score for s in per_language.values()]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance ** 0.5
    return mean_score, std_score


def format_results_table(
    results: Dict[str, TransformationResults],
    languages: List[str],
) -> str:
    """
    Format results as a markdown table.

    Args:
        results: Results per transformation type.
        languages: List of language codes.

    Returns:
        Markdown-formatted table.
    """
    lines = []
    lines.append("| Language | " + " | ".join(results.keys()) + " |")
    lines.append("|" + "---|" * (len(results) + 1))

    for lang in languages:
        row = [lang]
        for transform, res in results.items():
            score = res.per_language_scores[lang]
            row.append(f"{score.score:.3f} / {score.std:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
