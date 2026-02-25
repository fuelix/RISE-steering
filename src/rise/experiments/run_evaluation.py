"""
Run full evaluation suite for RISE paper.

This script runs RISE on the specified transformations and languages,
producing results suitable for the paper tables.

Usage:
    python -m rise.experiments.run_evaluation \
        --embedding-model sentence-transformers/LaBSE \
        --transformations negation conditionality politeness \
        --languages en es ja ar th ta zu \
        --output-dir results/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rise import RISE
from rise.baselines import MDV, Procrustes
from rise.evaluation import (
    compute_alignment_score,
    compute_cross_language_transfer,
    aggregate_results,
    format_results_table,
)
from rise.utils.config import ExperimentConfig
from rise.utils.reproducibility import set_seed, ExperimentLogger

logger = logging.getLogger(__name__)


TRANSFORMATION_TO_FILENAME = {
    "politeness": "polite",
}


def load_embeddings(
    data_dir: Path,
    transformation: str,
    language: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load neutral and transformed embeddings for a transformation/language pair.

    Reads JSONL pair files structured as {lang}/{transformation}_pairs.jsonl,
    where each line has "neutral" and "phenomenon" objects with "embedding" arrays.

    Args:
        data_dir: Directory containing embedding files.
        transformation: Transformation name (e.g., "negation").
        language: Language code (e.g., "en").

    Returns:
        Tuple of (neutral_embeddings, transformed_embeddings).
    """
    filename = TRANSFORMATION_TO_FILENAME.get(transformation, transformation)
    jsonl_path = data_dir / language / f"{filename}_pairs.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found for {transformation}/{language}. "
            f"Expected file: {jsonl_path}"
        )

    neutral_embeddings = []
    transformed_embeddings = []

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            neutral_embeddings.append(record["neutral"]["embedding"])
            transformed_embeddings.append(record["phenomenon"]["embedding"])

    neutral = torch.tensor(neutral_embeddings)
    transformed = torch.tensor(transformed_embeddings)

    # Normalize to unit sphere
    neutral = F.normalize(neutral, dim=1)
    transformed = F.normalize(transformed, dim=1)

    return neutral, transformed


def train_test_split(
    neutral: torch.Tensor,
    transformed: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split embeddings into train and test sets."""
    set_seed(seed)
    N = neutral.shape[0]
    indices = torch.randperm(N)
    n_train = int(N * train_ratio)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return (
        neutral[train_idx],
        transformed[train_idx],
        neutral[test_idx],
        transformed[test_idx],
    )


def evaluate_method(
    method: RISE,
    test_neutral: torch.Tensor,
    test_transformed: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate a trained method on test data.

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = []
    for i in range(test_neutral.shape[0]):
        result = method.transform(embedding=test_neutral[i])
        predictions.append(result.predicted_embedding)

    predictions = torch.stack(predictions)
    alignment = compute_alignment_score(predictions, test_transformed)

    return {
        "alignment_score": alignment.score,
        "n_test_samples": test_neutral.shape[0],
    }


def run_single_experiment(
    transformation: str,
    language: str,
    data_dir: Path,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run RISE on a single transformation/language pair.

    Returns:
        Dictionary mapping method names to their results.
    """
    logger.info(f"Running experiment: {transformation}/{language}")

    # Load data
    neutral, transformed = load_embeddings(data_dir, transformation, language)
    train_n, train_t, test_n, test_t = train_test_split(
        neutral, transformed, seed=seed
    )

    logger.info(f"  Train: {train_n.shape[0]}, Test: {test_n.shape[0]}")

    results = {}

    # RISE
    rise = RISE()
    rise.fit(neutral_embeddings=train_n, transformed_embeddings=train_t)
    results["RISE"] = evaluate_method(rise, test_n, test_t)
    logger.info(f"  RISE: {results['RISE']['alignment_score']:.4f}")

    # MDV baseline
    mdv = MDV()
    mdv.fit(train_n, train_t)
    results["MDV"] = evaluate_method(mdv, test_n, test_t)
    logger.info(f"  MDV: {results['MDV']['alignment_score']:.4f}")

    # Procrustes baseline
    procrustes = Procrustes()
    procrustes.fit(train_n, train_t)
    results["Procrustes"] = evaluate_method(procrustes, test_n, test_t)
    logger.info(f"  Procrustes: {results['Procrustes']['alignment_score']:.4f}")

    return results


def run_cross_language_experiment(
    transformation: str,
    languages: List[str],
    data_dir: Path,
    seed: int = 42,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Run cross-language transfer experiment.

    Train on each language, test on all languages.

    Returns:
        Dictionary mapping method names to transfer matrices.
    """
    logger.info(f"Running cross-language experiment: {transformation}")

    # Load all language data
    language_data = {}
    for lang in languages:
        try:
            neutral, transformed = load_embeddings(data_dir, transformation, lang)
            train_n, train_t, test_n, test_t = train_test_split(
                neutral, transformed, seed=seed
            )
            language_data[lang] = {
                "train_neutral": train_n,
                "train_transformed": train_t,
                "test_neutral": test_n,
                "test_transformed": test_t,
            }
        except FileNotFoundError as e:
            logger.warning(f"Skipping language {lang}: {e}")

    if len(language_data) < 2:
        logger.error("Need at least 2 languages for cross-language experiment")
        return {}

    results = {"RISE": {}, "MDV": {}, "Procrustes": {}}

    for train_lang in language_data:
        train_data = language_data[train_lang]

        # RISE
        rise = RISE()
        rise.fit(
            neutral_embeddings=train_data["train_neutral"],
            transformed_embeddings=train_data["train_transformed"],
        )

        # MDV baseline
        mdv = MDV()
        mdv.fit(
            train_data["train_neutral"],
            train_data["train_transformed"],
        )

        # Procrustes baseline
        procrustes = Procrustes()
        procrustes.fit(
            train_data["train_neutral"],
            train_data["train_transformed"],
        )

        # Test on all languages
        for test_lang in language_data:
            test_data = language_data[test_lang]

            for name, method in [("RISE", rise), ("MDV", mdv), ("Procrustes", procrustes)]:
                eval_results = evaluate_method(
                    method,
                    test_data["test_neutral"],
                    test_data["test_transformed"],
                )
                results[name][(train_lang, test_lang)] = eval_results[
                    "alignment_score"
                ]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RISE evaluation suite"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing embedding files",
    )
    parser.add_argument(
        "--transformations",
        nargs="+",
        default=["negation", "conditionality", "politeness"],
        help="Transformations to evaluate",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es", "ja", "ar", "th", "ta", "zu"],
        help="Languages to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cross-language",
        action="store_true",
        help="Run cross-language transfer experiments",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    all_results = {}

    # Run per-language experiments
    for transformation in args.transformations:
        all_results[transformation] = {}
        for language in args.languages:
            try:
                results = run_single_experiment(
                    transformation, language, args.data_dir, args.seed
                )
                all_results[transformation][language] = results
            except FileNotFoundError as e:
                logger.warning(f"Skipping {transformation}/{language}: {e}")

    # Save results
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        # Convert to JSON-serializable format
        json_results = {}
        for trans, lang_results in all_results.items():
            json_results[trans] = {}
            for lang, method_results in lang_results.items():
                json_results[trans][lang] = method_results
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Run cross-language experiments if requested
    if args.cross_language:
        cross_results = {}
        for transformation in args.transformations:
            cross_results[transformation] = run_cross_language_experiment(
                transformation, args.languages, args.data_dir, args.seed
            )

        cross_path = args.output_dir / "cross_language_results.json"
        with open(cross_path, "w") as f:
            # Convert tuple keys to strings for JSON
            json_cross = {}
            for trans, method_results in cross_results.items():
                json_cross[trans] = {}
                for method, transfer_matrix in method_results.items():
                    json_cross[trans][method] = {
                        f"{k[0]}->{k[1]}": v for k, v in transfer_matrix.items()
                    }
            json.dump(json_cross, f, indent=2)
        logger.info(f"Cross-language results saved to {cross_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    methods = ["RISE", "MDV", "Procrustes"]

    for transformation in args.transformations:
        if transformation not in all_results:
            continue
        print(f"\n{transformation.upper()}")
        print("-" * 50)

        header = "Language".ljust(10) + "".join(m.rjust(12) for m in methods)
        print(header)

        for language in args.languages:
            if language not in all_results[transformation]:
                continue
            lang_results = all_results[transformation][language]
            row = language.ljust(10)
            for method in methods:
                if method in lang_results:
                    score = lang_results[method]["alignment_score"]
                    row += f"{score:.4f}".rjust(12)
                else:
                    row += "N/A".rjust(12)
            print(row)


if __name__ == "__main__":
    main()
