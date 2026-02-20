"""
Run full evaluation suite for RISE.

This script runs RISE and all baselines on the specified transformations
and languages, producing results suitable for paper tables.

Usage:
    python -m rise.experiments.run_evaluation \
        --data-dir data/paper_embeddings \
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
from rise.baselines import ParkMethod, CAAMethod, HPRMethod, SteeringMethod
from rise.evaluation import (
    compute_alignment_score,
    compute_cross_language_transfer,
    aggregate_results,
    format_results_table,
)
from rise.utils.config import ExperimentConfig
from rise.utils.reproducibility import set_seed, ExperimentLogger

logger = logging.getLogger(__name__)

# Map transformation names to JSONL filenames
TRANSFORM_FILE_MAP = {
    "negation": "negation_pairs.jsonl",
    "conditionality": "conditionality_pairs.jsonl",
    "politeness": "polite_pairs.jsonl",
}


def load_embeddings(
    data_dir: Path,
    transformation: str,
    language: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load neutral and transformed embeddings for a transformation/language pair.

    Supports two data formats:
    - JSONL: ``data_dir/language/transformation_pairs.jsonl`` (HuggingFace dataset format)
    - PyTorch: ``data_dir/transformation_language_neutral.pt`` and ``*_transformed.pt``

    Args:
        data_dir: Directory containing embedding files.
        transformation: Transformation name (e.g., "negation").
        language: Language code (e.g., "en").

    Returns:
        Tuple of (neutral_embeddings, transformed_embeddings).
    """
    # Try JSONL format first (HuggingFace dataset layout)
    filename = TRANSFORM_FILE_MAP.get(transformation, f"{transformation}_pairs.jsonl")
    jsonl_path = data_dir / language / filename

    if jsonl_path.exists():
        return _load_jsonl_embeddings(jsonl_path)

    # Fall back to PyTorch tensor format
    neutral_path = data_dir / f"{transformation}_{language}_neutral.pt"
    transformed_path = data_dir / f"{transformation}_{language}_transformed.pt"

    if neutral_path.exists() and transformed_path.exists():
        return _load_pt_embeddings(neutral_path, transformed_path)

    raise FileNotFoundError(
        f"Embeddings not found for {transformation}/{language}. "
        f"Looked for JSONL at {jsonl_path} and PyTorch at {neutral_path}"
    )


def _load_jsonl_embeddings(filepath: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings from a JSONL file with neutral/phenomenon pairs."""
    neutral_embeddings = []
    transformed_embeddings = []

    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            neutral_embeddings.append(record["neutral"]["embedding"])
            transformed_embeddings.append(record["phenomenon"]["embedding"])

    neutral = torch.tensor(neutral_embeddings, dtype=torch.float32)
    transformed = torch.tensor(transformed_embeddings, dtype=torch.float32)

    neutral = F.normalize(neutral, dim=1)
    transformed = F.normalize(transformed, dim=1)

    return neutral, transformed


def _load_pt_embeddings(
    neutral_path: Path, transformed_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings from PyTorch tensor files."""
    neutral = torch.load(neutral_path)
    transformed = torch.load(transformed_path)

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
    method: SteeringMethod,
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
        result = method.transform(test_neutral[i])
        predictions.append(result.predicted_embedding)

    predictions = torch.stack(predictions)
    alignment = compute_alignment_score(predictions, test_transformed)

    return {
        "alignment_score": alignment,
        "n_test_samples": test_neutral.shape[0],
    }


def run_single_experiment(
    transformation: str,
    language: str,
    data_dir: Path,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run all methods on a single transformation/language pair.

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

    # Park
    park = ParkMethod(alpha=0.4)
    park.fit(train_n, train_t)
    results["Park"] = evaluate_method(park, test_n, test_t)

    # CAA
    caa = CAAMethod(strength=2.0)
    caa.fit(train_n, train_t)
    results["CAA"] = evaluate_method(caa, test_n, test_t)

    # HPR
    hpr = HPRMethod(n_reflections=2)
    hpr.fit(train_n, train_t)
    results["HPR"] = evaluate_method(hpr, test_n, test_t)

    for method_name, method_results in results.items():
        logger.info(f"  {method_name}: {method_results['alignment_score']:.4f}")

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

    methods = {
        "RISE": lambda: RISE(),
        "Park": lambda: ParkMethod(alpha=0.4),
        "CAA": lambda: CAAMethod(strength=2.0),
        "HPR": lambda: HPRMethod(n_reflections=2),
    }

    results = {name: {} for name in methods}

    for train_lang in language_data:
        train_data = language_data[train_lang]

        for method_name, method_factory in methods.items():
            method = method_factory()

            # Fit on training language
            if method_name == "RISE":
                method.fit(
                    neutral_embeddings=train_data["train_neutral"],
                    transformed_embeddings=train_data["train_transformed"],
                )
            else:
                method.fit(
                    train_data["train_neutral"],
                    train_data["train_transformed"],
                )

            # Test on all languages
            for test_lang in language_data:
                test_data = language_data[test_lang]
                eval_results = evaluate_method(
                    method,
                    test_data["test_neutral"],
                    test_data["test_transformed"],
                )
                results[method_name][(train_lang, test_lang)] = eval_results[
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

    for transformation in args.transformations:
        if transformation not in all_results:
            continue
        print(f"\n{transformation.upper()}")
        print("-" * 40)

        methods = ["RISE", "Park", "CAA", "HPR"]
        header = "Language".ljust(10) + "".join(m.rjust(10) for m in methods)
        print(header)

        for language in args.languages:
            if language not in all_results[transformation]:
                continue
            lang_results = all_results[transformation][language]
            row = language.ljust(10)
            for method in methods:
                if method in lang_results:
                    score = lang_results[method]["alignment_score"]
                    row += f"{score:.4f}".rjust(10)
                else:
                    row += "N/A".rjust(10)
            print(row)


if __name__ == "__main__":
    main()
