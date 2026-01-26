#!/usr/bin/env python3
"""
Verify RISE paper results by running on the original data.

This script loads the paper data from the helm repository and runs RISE
to verify we can replicate the reported alignment scores.

Expected results from ICLR 2026 paper:
- Negation:      0.864 mean (range 0.806-0.928)
- Conditionality: 0.832 mean (range 0.804-0.872)
- Politeness:    0.809 mean (range 0.770-0.883)

Usage:
    python scripts/verify_paper_results.py --data-dir ~/c/helm/eg_paper_data
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rise import RISE
from rise.baselines import ParkMethod, CAAMethod, HPRMethod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Expected results from ICLR 2026 paper
EXPECTED_RESULTS = {
    "negation": {
        "mean": 0.864,
        "std": 0.056,
        "by_language": {
            "ar": 0.806, "es": 0.831, "ja": 0.887,
            "ta": 0.911, "th": 0.880, "zu": 0.928, "en": 0.860
        }
    },
    "conditionality": {
        "mean": 0.832,
        "std": 0.062,
        "by_language": {
            "ar": 0.804, "es": 0.847, "ja": 0.872,
            "ta": 0.826, "th": 0.816, "zu": 0.847, "en": 0.840
        }
    },
    "politeness": {
        "mean": 0.809,
        "std": 0.073,
        "by_language": {
            "ar": 0.785, "es": 0.836, "ja": 0.883,
            "ta": 0.770, "th": 0.789, "zu": 0.772, "en": 0.820
        }
    }
}

LANGUAGES = ["en", "es", "ja", "ar", "th", "ta", "zu"]
TRANSFORMATIONS = ["negation", "conditionality", "politeness"]

# Map file naming conventions
TRANSFORM_FILE_MAP = {
    "negation": "negation_pairs.jsonl",
    "conditionality": "conditionality_pairs.jsonl",
    "politeness": "polite_pairs.jsonl",
}


def load_pairs(data_dir: Path, language: str, transformation: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load neutral and transformed embeddings from JSONL file.

    Returns:
        Tuple of (neutral_embeddings, transformed_embeddings) as tensors.
    """
    filename = TRANSFORM_FILE_MAP.get(transformation, f"{transformation}_pairs.jsonl")
    filepath = data_dir / language / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    neutral_embeddings = []
    transformed_embeddings = []

    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            neutral_emb = record["neutral"]["embedding"]
            transformed_emb = record["phenomenon"]["embedding"]

            neutral_embeddings.append(neutral_emb)
            transformed_embeddings.append(transformed_emb)

    neutral = torch.tensor(neutral_embeddings, dtype=torch.float32)
    transformed = torch.tensor(transformed_embeddings, dtype=torch.float32)

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
    """Split data into train and test sets."""
    torch.manual_seed(seed)
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


def compute_alignment_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean cosine similarity between predictions and targets."""
    predictions = F.normalize(predictions, dim=1)
    targets = F.normalize(targets, dim=1)
    similarities = (predictions * targets).sum(dim=1)
    return similarities.mean().item()


def evaluate_rise(
    train_neutral: torch.Tensor,
    train_transformed: torch.Tensor,
    test_neutral: torch.Tensor,
    test_transformed: torch.Tensor,
) -> float:
    """Train RISE and evaluate on test set."""
    rise = RISE()
    rise.fit(
        neutral_embeddings=train_neutral,
        transformed_embeddings=train_transformed,
    )

    predictions = []
    for i in range(test_neutral.shape[0]):
        result = rise.transform(embedding=test_neutral[i])
        predictions.append(result.predicted_embedding)

    predictions = torch.stack(predictions)
    return compute_alignment_score(predictions, test_transformed)


def run_verification(data_dir: Path, seed: int = 42) -> Dict:
    """
    Run full verification across all transformations and languages.

    Returns:
        Dictionary with results and comparison to expected values.
    """
    results = {}

    for transformation in TRANSFORMATIONS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {transformation.upper()}")
        logger.info(f"{'='*60}")

        results[transformation] = {"by_language": {}, "scores": []}

        for language in LANGUAGES:
            try:
                # Load data
                neutral, transformed = load_pairs(data_dir, language, transformation)
                logger.info(f"  {language}: Loaded {neutral.shape[0]} pairs, dim={neutral.shape[1]}")

                # Split
                train_n, train_t, test_n, test_t = train_test_split(
                    neutral, transformed, seed=seed
                )

                # Evaluate RISE
                score = evaluate_rise(train_n, train_t, test_n, test_t)

                results[transformation]["by_language"][language] = score
                results[transformation]["scores"].append(score)

                # Compare to expected
                expected = EXPECTED_RESULTS[transformation]["by_language"].get(language)
                diff = score - expected if expected else 0
                status = "OK" if abs(diff) < 0.05 else "DIFF"

                logger.info(f"  {language}: {score:.4f} (expected: {expected:.4f}, diff: {diff:+.4f}) [{status}]")

            except FileNotFoundError as e:
                logger.warning(f"  {language}: Skipped - {e}")
            except Exception as e:
                logger.error(f"  {language}: Error - {e}")

        # Compute aggregate stats
        if results[transformation]["scores"]:
            scores = results[transformation]["scores"]
            results[transformation]["mean"] = sum(scores) / len(scores)
            results[transformation]["std"] = (
                sum((s - results[transformation]["mean"])**2 for s in scores) / len(scores)
            ) ** 0.5

            expected_mean = EXPECTED_RESULTS[transformation]["mean"]
            diff = results[transformation]["mean"] - expected_mean

            logger.info(f"\n  AGGREGATE: {results[transformation]['mean']:.4f} +/- {results[transformation]['std']:.4f}")
            logger.info(f"  EXPECTED:  {expected_mean:.4f} +/- {EXPECTED_RESULTS[transformation]['std']:.4f}")
            logger.info(f"  DIFFERENCE: {diff:+.4f}")

    return results


def print_summary(results: Dict) -> None:
    """Print a summary comparison table."""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    print(f"\n{'Transformation':<15} {'Obtained':<12} {'Expected':<12} {'Diff':<10} {'Status':<10}")
    print("-"*60)

    all_pass = True
    for transformation in TRANSFORMATIONS:
        if transformation not in results or "mean" not in results[transformation]:
            continue

        obtained = results[transformation]["mean"]
        expected = EXPECTED_RESULTS[transformation]["mean"]
        diff = obtained - expected

        # Allow 5% tolerance for replication
        status = "PASS" if abs(diff) < 0.05 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{transformation:<15} {obtained:<12.4f} {expected:<12.4f} {diff:+<10.4f} {status:<10}")

    print("-"*60)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Verify RISE paper results")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data/paper_embeddings",
        help="Path to paper data directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Random seed: {args.seed}")

    results = run_verification(args.data_dir, args.seed)
    print_summary(results)


if __name__ == "__main__":
    main()
