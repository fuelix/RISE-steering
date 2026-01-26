"""
Visualization utilities for RISE experiments.

This module provides plotting functions for generating figures
similar to those in the ICLR paper.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available; visualization functions will not work")


def plot_cross_language_heatmap(
    transfer_matrix: Dict[Tuple[str, str], float],
    languages: List[str],
    title: str = "Cross-Language Transfer",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "YlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Optional[object]:
    """
    Plot a heatmap of cross-language transfer scores.

    Args:
        transfer_matrix: Dictionary mapping (source, target) pairs to scores.
        languages: List of language codes in order.
        title: Plot title.
        save_path: Path to save the figure (optional).
        figsize: Figure size in inches.
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.

    Returns:
        matplotlib Figure object if matplotlib available, else None.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for visualization")
        return None

    n_langs = len(languages)
    matrix = [[0.0] * n_langs for _ in range(n_langs)]

    # Fill matrix
    for i, src in enumerate(languages):
        for j, tgt in enumerate(languages):
            key = (src, tgt)
            if key in transfer_matrix:
                matrix[i][j] = transfer_matrix[key]
            elif (tgt, src) in transfer_matrix:
                matrix[i][j] = transfer_matrix[(tgt, src)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    # Labels
    ax.set_xticks(range(n_langs))
    ax.set_yticks(range(n_langs))
    ax.set_xticklabels(languages)
    ax.set_yticklabels(languages)
    ax.set_xlabel("Test Language")
    ax.set_ylabel("Train Language")
    ax.set_title(title)

    # Add text annotations
    for i in range(n_langs):
        for j in range(n_langs):
            text = ax.text(j, i, f"{matrix[i][j]:.3f}",
                          ha="center", va="center", color="black", fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Alignment Score")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_centroid_similarity(
    similarities: Dict[Tuple[str, str], float],
    languages: List[str],
    title: str = "Centroid Similarity",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[object]:
    """
    Plot a heatmap of centroid similarities across languages.

    Args:
        similarities: Dictionary mapping language pairs to similarity scores.
        languages: List of language codes.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object if available.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for visualization")
        return None

    n_langs = len(languages)
    matrix = [[0.0] * n_langs for _ in range(n_langs)]

    for i, l1 in enumerate(languages):
        for j, l2 in enumerate(languages):
            if i == j:
                matrix[i][j] = 1.0
            elif (l1, l2) in similarities:
                matrix[i][j] = similarities[(l1, l2)]
                matrix[j][i] = similarities[(l1, l2)]
            elif (l2, l1) in similarities:
                matrix[i][j] = similarities[(l2, l1)]
                matrix[j][i] = similarities[(l2, l1)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="YlGn", vmin=0.8, vmax=1.0)

    ax.set_xticks(range(n_langs))
    ax.set_yticks(range(n_langs))
    ax.set_xticklabels(languages)
    ax.set_yticklabels(languages)
    ax.set_title(title)

    for i in range(n_langs):
        for j in range(n_langs):
            ax.text(j, i, f"{matrix[i][j]:.3f}",
                   ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_transformation_comparison(
    results: Dict[str, Dict[str, float]],
    languages: List[str],
    title: str = "Transformation Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[object]:
    """
    Plot a grouped bar chart comparing transformations across languages.

    Args:
        results: Dict mapping transformation names to {language: score} dicts.
        languages: List of language codes.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object if available.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for visualization")
        return None

    import numpy as np

    transformations = list(results.keys())
    n_groups = len(languages)
    n_bars = len(transformations)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, transform in enumerate(transformations):
        scores = [results[transform].get(lang, 0) for lang in languages]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + offset, scores, bar_width, label=transform, color=colors[i % len(colors)])

    ax.set_xlabel("Language")
    ax.set_ylabel("Alignment Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.set_ylim(0.7, 1.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def create_paper_figures(
    results_dir: Path,
    output_dir: Path,
    languages: List[str] = None,
) -> None:
    """
    Generate all figures for the ICLR paper.

    Args:
        results_dir: Directory containing experiment results.
        output_dir: Directory to save figures.
        languages: List of languages (default: paper languages).
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required to generate figures")
        return

    if languages is None:
        languages = ["en", "ja", "es", "ar", "th", "ta", "zu"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating paper figures in {output_dir}")

    # Load results and generate figures
    # This would be implemented based on actual result file format

    logger.info("Figure generation complete")
