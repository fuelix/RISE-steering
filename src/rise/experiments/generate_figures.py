"""
Generate paper figures from experiment results.

This script produces publication-ready figures for the ICLR paper.

Usage:
    python -m rise.experiments.generate_figures \
        --results-dir results/ \
        --output-dir figures/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Import visualization (may not be available without matplotlib)
try:
    from rise.evaluation import (
        plot_cross_language_heatmap,
        plot_centroid_similarity,
        plot_transformation_comparison,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    logger.warning("Visualization not available - install matplotlib")


def load_results(results_dir: Path) -> Dict:
    """Load experiment results from JSON files."""
    results = {}

    # Load main results
    main_path = results_dir / "results.json"
    if main_path.exists():
        with open(main_path) as f:
            results["main"] = json.load(f)

    # Load cross-language results
    cross_path = results_dir / "cross_language_results.json"
    if cross_path.exists():
        with open(cross_path) as f:
            cross_data = json.load(f)
            # Convert string keys back to tuples
            results["cross_language"] = {}
            for trans, method_results in cross_data.items():
                results["cross_language"][trans] = {}
                for method, transfers in method_results.items():
                    results["cross_language"][trans][method] = {
                        tuple(k.split("->")):
                        v for k, v in transfers.items()
                    }

    return results


def generate_main_comparison_figure(
    results: Dict,
    output_dir: Path,
    languages: List[str],
) -> None:
    """Generate main comparison bar chart (Figure 2 in paper)."""
    if not HAS_VIZ:
        logger.error("matplotlib required for visualization")
        return

    main_results = results.get("main", {})

    for transformation in main_results:
        # Prepare data for plotting
        method_scores = {"RISE": {}}

        for lang, lang_results in main_results[transformation].items():
            if lang in languages:
                for method in method_scores:
                    if method in lang_results:
                        method_scores[method][lang] = lang_results[method][
                            "alignment_score"
                        ]

        fig = plot_transformation_comparison(
            method_scores,
            languages=[l for l in languages if l in method_scores["RISE"]],
            title=f"{transformation.title()} Transformation",
            save_path=output_dir / f"{transformation}_comparison.pdf",
        )

        if fig:
            logger.info(f"Generated {transformation} comparison figure")


def generate_cross_language_figures(
    results: Dict,
    output_dir: Path,
    languages: List[str],
) -> None:
    """Generate cross-language transfer heatmaps (Figure 3 in paper)."""
    if not HAS_VIZ:
        logger.error("matplotlib required for visualization")
        return

    cross_results = results.get("cross_language", {})

    for transformation in cross_results:
        for method, transfer_matrix in cross_results[transformation].items():
            # Filter to available languages
            available_langs = set()
            for src, tgt in transfer_matrix.keys():
                available_langs.add(src)
                available_langs.add(tgt)
            filtered_langs = [l for l in languages if l in available_langs]

            if len(filtered_langs) < 2:
                continue

            fig = plot_cross_language_heatmap(
                transfer_matrix,
                filtered_langs,
                title=f"{method} - {transformation.title()} Transfer",
                save_path=output_dir / f"{transformation}_{method}_transfer.pdf",
            )

            if fig:
                logger.info(f"Generated {method} {transformation} transfer heatmap")


def generate_summary_table(
    results: Dict,
    output_dir: Path,
    languages: List[str],
) -> None:
    """Generate LaTeX summary table for paper."""
    main_results = results.get("main", {})

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Alignment scores across transformations and languages}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "c" * len(languages) + r"}",
        r"\toprule",
        "Method & " + " & ".join(languages) + r" \\",
        r"\midrule",
    ]

    for transformation in main_results:
        latex_lines.append(r"\multicolumn{" + str(len(languages) + 1) + r"}{l}{\textbf{" + transformation.title() + r"}} \\")

        for method in ["RISE"]:
            scores = []
            for lang in languages:
                if lang in main_results[transformation]:
                    lang_results = main_results[transformation][lang]
                    if method in lang_results:
                        score = lang_results[method]["alignment_score"]
                        scores.append(f"{score:.3f}")
                    else:
                        scores.append("--")
                else:
                    scores.append("--")
            latex_lines.append(method + " & " + " & ".join(scores) + r" \\")

        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_path = output_dir / "results_table.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Generated LaTeX table: {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RISE paper figures"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es", "ja", "ar", "th", "ta", "zu"],
        help="Languages to include in figures",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Output format for figures",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results_dir)

    if not results:
        logger.error(f"No results found in {args.results_dir}")
        return

    # Generate figures
    generate_main_comparison_figure(results, args.output_dir, args.languages)
    generate_cross_language_figures(results, args.output_dir, args.languages)
    generate_summary_table(results, args.output_dir, args.languages)

    logger.info(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
