"""
Evaluation metrics and visualization for RISE.

This module contains alignment score computation, cross-language
transfer evaluation, and plotting utilities.
"""

from .metrics import (
    compute_alignment_score,
    compute_cross_language_transfer,
    compute_centroid_similarity,
    aggregate_results,
    format_results_table,
    TransformationResults,
)
from .visualization import (
    plot_cross_language_heatmap,
    plot_centroid_similarity,
    plot_transformation_comparison,
    create_paper_figures,
)

__all__ = [
    # Metrics
    "compute_alignment_score",
    "compute_cross_language_transfer",
    "compute_centroid_similarity",
    "aggregate_results",
    "format_results_table",
    "TransformationResults",
    # Visualization
    "plot_cross_language_heatmap",
    "plot_centroid_similarity",
    "plot_transformation_comparison",
    "create_paper_figures",
]
