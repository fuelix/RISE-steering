"""
Evaluation metrics for RISE.

This module contains alignment score computation, cross-language
transfer evaluation, and result aggregation utilities.
"""

from .metrics import (
    compute_alignment_score,
    compute_cross_language_transfer,
    compute_centroid_similarity,
    aggregate_results,
    format_results_table,
    TransformationResults,
)

__all__ = [
    "compute_alignment_score",
    "compute_cross_language_transfer",
    "compute_centroid_similarity",
    "aggregate_results",
    "format_results_table",
    "TransformationResults",
]
