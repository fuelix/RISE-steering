#!/usr/bin/env python3
"""
Verify RISE paper results against published numbers (ICLR 2026).

Compares code output against every number in the paper:

  1. Cell-level — all 441 individual (train_lang, test_lang) scores
     from Figures 2, 3, 5, 6, 7 (3 models x 3 phenomena x 7x7)
  2. Table 2 — Per-model Synthetic Multilingual averages
  3. Section 6.1 — Per-phenomenon cross-model aggregates

Usage:
    python scripts/verify_paper_results.py
    python scripts/verify_paper_results.py --seed 42
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rise.experiments.run_evaluation import run_cross_language_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Paper heatmap values (Figures 2, 3, 5, 6, 7) ─────────────────────────
# Each matrix is {source_lang: [scores for target langs in LANG_ORDER]}.
# Column order: AR, EN, ES, JA, TA, TH, ZU

LANG_ORDER = ["ar", "en", "es", "ja", "ta", "th", "zu"]

PAPER_HEATMAPS = {
    "text-embedding-3-large": {
        "negation": {
            "ar": [0.802, 0.731, 0.804, 0.848, 0.848, 0.839, 0.820],
            "en": [0.775, 0.763, 0.807, 0.852, 0.848, 0.839, 0.832],
            "es": [0.773, 0.733, 0.823, 0.841, 0.832, 0.825, 0.804],
            "ja": [0.768, 0.726, 0.793, 0.860, 0.844, 0.837, 0.798],
            "ta": [0.764, 0.710, 0.782, 0.848, 0.890, 0.850, 0.837],
            "th": [0.781, 0.729, 0.800, 0.864, 0.875, 0.875, 0.846],
            "zu": [0.754, 0.703, 0.779, 0.844, 0.873, 0.851, 0.918],
        },
        "conditionality": {
            "ar": [0.785, 0.777, 0.806, 0.830, 0.757, 0.763, 0.746],
            "en": [0.751, 0.812, 0.816, 0.836, 0.748, 0.758, 0.746],
            "es": [0.752, 0.786, 0.837, 0.827, 0.741, 0.751, 0.739],
            "ja": [0.746, 0.779, 0.802, 0.860, 0.750, 0.762, 0.734],
            "ta": [0.734, 0.751, 0.774, 0.813, 0.811, 0.747, 0.753],
            "th": [0.754, 0.777, 0.801, 0.840, 0.762, 0.804, 0.749],
            "zu": [0.683, 0.707, 0.735, 0.758, 0.715, 0.692, 0.836],
        },
        "politeness": {
            "ar": [0.777, 0.757, 0.791, 0.833, 0.653, 0.729, 0.651],
            "en": [0.732, 0.795, 0.793, 0.819, 0.646, 0.721, 0.643],
            "es": [0.739, 0.769, 0.828, 0.827, 0.661, 0.727, 0.655],
            "ja": [0.728, 0.740, 0.769, 0.855, 0.623, 0.709, 0.622],
            "ta": [0.671, 0.691, 0.737, 0.764, 0.770, 0.692, 0.661],
            "th": [0.729, 0.748, 0.783, 0.824, 0.671, 0.784, 0.665],
            "zu": [0.663, 0.683, 0.724, 0.749, 0.665, 0.679, 0.765],
        },
    },
    "mBERT": {
        "negation": {
            "ar": [0.858, 0.644, 0.789, 0.637, 0.812, 0.778, 0.752],
            "en": [0.735, 0.887, 0.781, 0.727, 0.748, 0.672, 0.692],
            "es": [0.804, 0.683, 0.862, 0.677, 0.802, 0.749, 0.755],
            "ja": [0.689, 0.687, 0.720, 0.853, 0.717, 0.674, 0.633],
            "ta": [0.802, 0.641, 0.778, 0.657, 0.862, 0.765, 0.754],
            "th": [0.787, 0.638, 0.755, 0.655, 0.785, 0.851, 0.734],
            "zu": [0.756, 0.631, 0.746, 0.592, 0.762, 0.722, 0.861],
        },
        "conditionality": {
            "ar": [0.848, 0.653, 0.792, 0.662, 0.810, 0.712, 0.733],
            "en": [0.744, 0.869, 0.784, 0.719, 0.739, 0.622, 0.688],
            "es": [0.799, 0.699, 0.864, 0.696, 0.804, 0.687, 0.742],
            "ja": [0.674, 0.664, 0.699, 0.850, 0.695, 0.608, 0.607],
            "ta": [0.787, 0.645, 0.774, 0.673, 0.867, 0.696, 0.733],
            "th": [0.777, 0.660, 0.761, 0.686, 0.781, 0.788, 0.708],
            "zu": [0.723, 0.614, 0.727, 0.591, 0.744, 0.635, 0.852],
        },
        "politeness": {
            "ar": [0.845, 0.599, 0.778, 0.658, 0.809, 0.476, 0.704],
            "en": [0.657, 0.823, 0.679, 0.651, 0.639, 0.365, 0.591],
            "es": [0.784, 0.639, 0.853, 0.681, 0.801, 0.449, 0.712],
            "ja": [0.630, 0.585, 0.641, 0.858, 0.632, 0.412, 0.543],
            "ta": [0.769, 0.570, 0.756, 0.637, 0.864, 0.449, 0.700],
            "th": [0.682, 0.562, 0.658, 0.623, 0.675, 0.658, 0.624],
            "zu": [0.681, 0.542, 0.687, 0.551, 0.710, 0.417, 0.853],
        },
    },
    "bge-m3": {
        "negation": {
            "ar": [0.767, 0.789, 0.784, 0.783, 0.783, 0.779, 0.710],
            "en": [0.743, 0.817, 0.784, 0.785, 0.771, 0.790, 0.693],
            "es": [0.751, 0.787, 0.799, 0.783, 0.777, 0.771, 0.687],
            "ja": [0.740, 0.780, 0.776, 0.807, 0.786, 0.792, 0.686],
            "ta": [0.738, 0.763, 0.765, 0.783, 0.796, 0.775, 0.695],
            "th": [0.728, 0.780, 0.760, 0.786, 0.767, 0.818, 0.689],
            "zu": [0.724, 0.753, 0.738, 0.741, 0.749, 0.750, 0.755],
        },
        "conditionality": {
            "ar": [0.783, 0.845, 0.814, 0.806, 0.795, 0.800, 0.750],
            "en": [0.746, 0.863, 0.807, 0.798, 0.780, 0.783, 0.719],
            "es": [0.766, 0.854, 0.835, 0.807, 0.791, 0.798, 0.734],
            "ja": [0.745, 0.832, 0.795, 0.831, 0.798, 0.801, 0.725],
            "ta": [0.753, 0.828, 0.795, 0.812, 0.819, 0.802, 0.747],
            "th": [0.749, 0.833, 0.795, 0.812, 0.797, 0.835, 0.734],
            "zu": [0.737, 0.788, 0.765, 0.765, 0.771, 0.766, 0.785],
        },
        "politeness": {
            "ar": [0.811, 0.849, 0.814, 0.818, 0.808, 0.818, 0.746],
            "en": [0.782, 0.865, 0.804, 0.798, 0.784, 0.799, 0.712],
            "es": [0.798, 0.860, 0.834, 0.818, 0.800, 0.816, 0.723],
            "ja": [0.779, 0.825, 0.793, 0.837, 0.798, 0.806, 0.710],
            "ta": [0.788, 0.831, 0.796, 0.813, 0.829, 0.812, 0.747],
            "th": [0.785, 0.836, 0.801, 0.813, 0.801, 0.844, 0.736],
            "zu": [0.743, 0.775, 0.740, 0.746, 0.762, 0.765, 0.792],
        },
    },
}

# Table 2: RISE Performance, Synthetic Multilingual column
PAPER_TABLE_2 = {
    "text-embedding-3-large": 0.771,
    "bge-m3": 0.782,
    "mBERT": 0.709,
}

# Section 6.1: Per-phenomenon cross-model aggregates
PAPER_SECTION_6_1 = {
    "negation": 0.788,
    "conditionality": 0.780,
    "politeness": 0.762,
}


# ── Configuration ──────────────────────────────────────────────────────────

MODEL_DATA_DIRS = {
    "text-embedding-3-large": Path("data/paper_embeddings"),
    "bge-m3": Path("data/bge-m3"),
    "mBERT": Path("data/mbert"),
}

LANGUAGES = ["ar", "en", "es", "ja", "ta", "th", "zu"]
PHENOMENA = ["negation", "conditionality", "politeness"]


# ── Core logic ─────────────────────────────────────────────────────────────

def run_all_matrices(
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[Tuple[str, str], float]]]:
    """
    Run 7x7 cross-language transfer for all models x phenomena.

    Returns:
        Nested dict: model -> phenomenon -> {(train_lang, test_lang): score}
    """
    base_dir = Path(__file__).parent.parent
    all_results = {}

    for model, rel_data_dir in MODEL_DATA_DIRS.items():
        data_dir = base_dir / rel_data_dir
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model}")
        logger.info(f"Data:  {data_dir}")
        logger.info(f"{'='*60}")

        all_results[model] = {}

        for phenomenon in PHENOMENA:
            logger.info(f"  Running {phenomenon} 7x7 matrix...")
            t0 = time.time()

            cross_results = run_cross_language_experiment(
                transformation=phenomenon,
                languages=LANGUAGES,
                data_dir=data_dir,
                seed=seed,
            )

            matrix = cross_results.get("RISE", {})
            all_results[model][phenomenon] = matrix

            n_pairs = len(matrix)
            elapsed = time.time() - t0
            if n_pairs > 0:
                mean = sum(matrix.values()) / n_pairs
                logger.info(
                    f"  {phenomenon}: {n_pairs} pairs, mean={mean:.4f} "
                    f"({elapsed:.1f}s)"
                )
            else:
                logger.warning(f"  {phenomenon}: no results!")

    return all_results


def compute_table_2(
    all_results: Dict[str, Dict[str, Dict[Tuple[str, str], float]]],
) -> Dict[str, float]:
    """Per-model average across all phenomena and all 7x7 transfer pairs."""
    table_2 = {}
    for model in MODEL_DATA_DIRS:
        if model not in all_results:
            continue
        scores = []
        for phenomenon in PHENOMENA:
            matrix = all_results[model].get(phenomenon, {})
            scores.extend(matrix.values())
        if scores:
            table_2[model] = sum(scores) / len(scores)
    return table_2


def compute_section_6_1(
    all_results: Dict[str, Dict[str, Dict[Tuple[str, str], float]]],
) -> Dict[str, float]:
    """Per-phenomenon average across all models and all 7x7 transfer pairs."""
    section_6_1 = {}
    for phenomenon in PHENOMENA:
        scores = []
        for model in MODEL_DATA_DIRS:
            if model not in all_results:
                continue
            matrix = all_results[model].get(phenomenon, {})
            scores.extend(matrix.values())
        if scores:
            section_6_1[phenomenon] = sum(scores) / len(scores)
    return section_6_1


# ── Output formatting ──────────────────────────────────────────────────────

def print_cell_comparison(
    all_results: Dict[str, Dict[str, Dict[Tuple[str, str], float]]],
) -> None:
    """Print per-cell comparison of obtained vs paper heatmap values."""
    all_diffs = []

    for model in MODEL_DATA_DIRS:
        if model not in PAPER_HEATMAPS or model not in all_results:
            continue

        for phenomenon in PHENOMENA:
            paper_matrix = PAPER_HEATMAPS[model].get(phenomenon, {})
            obtained_matrix = all_results[model].get(phenomenon, {})

            if not paper_matrix or not obtained_matrix:
                continue

            print(f"\n  {model} / {phenomenon}")
            print(f"  {'':>5}", end="")
            for tgt in LANG_ORDER:
                print(f" {tgt:>7}", end="")
            print()

            for i, src in enumerate(LANG_ORDER):
                paper_row = paper_matrix[src]
                print(f"  {src:>5}", end="")

                for j, tgt in enumerate(LANG_ORDER):
                    obt = obtained_matrix.get((src, tgt))
                    paper_val = paper_row[j]
                    if obt is not None:
                        diff = obt - paper_val
                        all_diffs.append(diff)
                        print(f" {diff:>+7.3f}", end="")
                    else:
                        print(f" {'N/A':>7}", end="")
                print()

    if all_diffs:
        mean_diff = sum(all_diffs) / len(all_diffs)
        abs_diffs = [abs(d) for d in all_diffs]
        mean_abs = sum(abs_diffs) / len(abs_diffs)
        max_abs = max(abs_diffs)
        print(f"\n  Cell diff summary ({len(all_diffs)} cells):")
        print(f"    Mean diff:     {mean_diff:+.4f}")
        print(f"    Mean |diff|:   {mean_abs:.4f}")
        print(f"    Max  |diff|:   {max_abs:.4f}")


def print_comparison_table(
    obtained: Dict[str, float],
    expected: Dict[str, float],
    header: str,
    label_header: str,
) -> None:
    """Print comparison table showing obtained vs paper values."""
    print(f"\n{'='*58}")
    print(f"  {header}")
    print(f"{'='*58}")
    print(f"  {label_header:<25} {'Obtained':>10} {'Paper':>10} {'Diff':>10}")
    print(f"  {'-'*55}")

    for key in expected:
        if key not in obtained:
            print(f"  {key:<25} {'N/A':>10} {expected[key]:>10.3f} {'':>10}")
            continue

        obt = obtained[key]
        exp = expected[key]
        diff = obt - exp
        print(f"  {key:<25} {obt:>10.4f} {exp:>10.3f} {diff:>+10.4f}")

    print(f"  {'-'*55}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify RISE paper results against published numbers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("\nRISE Paper Verification")
    print("Runs full 7x7 cross-language transfer for all models x phenomena\n")

    # Run all 7x7 matrices (441 evaluations total)
    t0 = time.time()
    all_results = run_all_matrices(seed=args.seed)
    elapsed = time.time() - t0
    logger.info(f"\nAll matrices completed in {elapsed:.1f}s")

    # Compute aggregates
    table_2 = compute_table_2(all_results)
    section_6_1 = compute_section_6_1(all_results)

    # Cell-level comparison (obtained - paper, per cell)
    print(f"\n{'='*68}")
    print(f"  CELL-LEVEL DIFFS (obtained - paper) for each 7x7 matrix")
    print(f"{'='*68}")
    print_cell_comparison(all_results)

    # Table 2
    print_comparison_table(
        table_2,
        PAPER_TABLE_2,
        header="TABLE 2 (Synthetic Multilingual)",
        label_header="Model",
    )

    # Section 6.1
    print_comparison_table(
        section_6_1,
        PAPER_SECTION_6_1,
        header="SECTION 6.1 (Per-Phenomenon Aggregates)",
        label_header="Phenomenon",
    )

    print(f"\n  Total runtime: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
