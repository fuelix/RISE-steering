#!/usr/bin/env python3
"""
Reproduce Table 9 (Appendix G): Downstream negation classification.

Uses MDV and RISE as binary classifiers to detect whether a sentence
is negated. For each method, learns a 1-D projection direction from
training pairs, then finds the optimal F1 threshold on training data.

Usage:
    python scripts/run_classification.py
    python scripts/run_classification.py --data-dir data/text-embedding-3-large
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from rise import RISE
from rise.baselines import MDV
from rise.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_training_pairs(data_dir: Path):
    """Load negation training pairs and return (neutral, transformed) embedding tensors."""
    pairs_path = data_dir / "en" / "negation_pairs.jsonl"
    neutral, transformed = [], []
    with open(pairs_path) as f:
        for line in f:
            record = json.loads(line)
            neutral.append(record["neutral"]["embedding"])
            transformed.append(record["phenomenon"]["embedding"])
    neutral = F.normalize(torch.tensor(neutral), dim=1)
    transformed = F.normalize(torch.tensor(transformed), dim=1)
    return neutral, transformed


def load_test_data(test_path: Path):
    """Load classification test set and return (embeddings, labels)."""
    embeddings, labels = [], []
    with open(test_path) as f:
        for line in f:
            record = json.loads(line)
            embeddings.append(record["embedding"])
            labels.append(record["label"])
    embeddings = F.normalize(torch.tensor(embeddings), dim=1)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels


def find_optimal_threshold(scores: torch.Tensor, labels: torch.Tensor):
    """Find the threshold that maximizes F1 score."""
    sorted_scores, _ = scores.sort()
    best_f1, best_thresh = 0.0, 0.0

    # Evaluate at each unique score as a candidate threshold
    candidates = sorted_scores.unique()
    for thresh in candidates:
        preds = (scores >= thresh).long()
        f1 = f1_score(labels.numpy(), preds.numpy(), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh.item()

    return best_thresh


def classify_mdv(
    train_neutral: torch.Tensor,
    train_transformed: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
):
    """
    MDV classification: project embeddings onto the mean difference vector.

    Higher projection = more likely negated (label=1).
    """
    mdv = MDV()
    mdv.fit(train_neutral, train_transformed)
    prototype = F.normalize(mdv.prototype, dim=0)

    # Score = dot product with normalized prototype direction
    train_all = torch.cat([train_neutral, train_transformed], dim=0)
    train_labels = torch.cat([
        torch.zeros(train_neutral.shape[0], dtype=torch.long),
        torch.ones(train_transformed.shape[0], dtype=torch.long),
    ])
    train_scores = train_all @ prototype

    # Find optimal threshold on training data
    threshold = find_optimal_threshold(train_scores, train_labels)

    # Apply to test set
    test_scores = test_embeddings @ prototype
    test_preds = (test_scores >= threshold).long()

    return compute_metrics(test_labels, test_preds)


def classify_rise(
    train_neutral: torch.Tensor,
    train_transformed: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
):
    """
    RISE classification: project embeddings onto the canonicalized prototype.

    The RISE prototype is learned via Riemannian averaging in the canonical
    frame (tangent space at e1), producing a cleaner direction than MDV's
    Euclidean average. We use this direction to score embeddings: higher
    projection onto the prototype = more likely negated.
    """
    rise = RISE()
    rise.fit(neutral_embeddings=train_neutral, transformed_embeddings=train_transformed)
    prototype = rise.prototype.prototype  # tangent vector at e1

    # Use the normalized prototype as the classification direction.
    # The prototype lies in T_{e1} (orthogonal to e1) and captures the
    # canonicalized negation direction, which serves as a discriminative
    # projection axis in the ambient space.
    direction = F.normalize(prototype, dim=0)

    # Score training data to find threshold
    train_all = torch.cat([train_neutral, train_transformed], dim=0)
    train_labels = torch.cat([
        torch.zeros(train_neutral.shape[0], dtype=torch.long),
        torch.ones(train_transformed.shape[0], dtype=torch.long),
    ])
    train_scores = train_all @ direction

    threshold = find_optimal_threshold(train_scores, train_labels)

    # Apply to test set
    test_scores = test_embeddings @ direction
    test_preds = (test_scores >= threshold).long()

    return compute_metrics(test_labels, test_preds)


def compute_metrics(labels: torch.Tensor, preds: torch.Tensor):
    """Compute accuracy, precision, recall, F1."""
    y_true = labels.numpy()
    y_pred = preds.numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 9: Downstream negation classification"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "text-embedding-3-large",
        help="Directory containing embedding pair files",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "negation_classification_test" / "negation_test_embedded.jsonl",
        help="Path to classification test set",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    set_seed(args.seed)

    # Load data
    logger.info("Loading training pairs...")
    train_neutral, train_transformed = load_training_pairs(args.data_dir)
    logger.info(f"  {train_neutral.shape[0]} training pairs, dim={train_neutral.shape[1]}")

    logger.info("Loading test data...")
    test_embeddings, test_labels = load_test_data(args.test_file)
    logger.info(
        f"  {test_embeddings.shape[0]} test samples "
        f"({test_labels.sum().item()} positive, {(~test_labels.bool()).sum().item()} negative)"
    )

    # Run classifiers
    logger.info("Running MDV classifier...")
    mdv_results = classify_mdv(train_neutral, train_transformed, test_embeddings, test_labels)

    logger.info("Running RISE classifier...")
    rise_results = classify_rise(train_neutral, train_transformed, test_embeddings, test_labels)

    # Print Table 9
    print()
    print("=" * 60)
    print("TABLE 9: DOWNSTREAM NEGATION CLASSIFICATION")
    print("=" * 60)
    print(f"{'Method':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 54)
    for name, results in [("MDV", mdv_results), ("RISE", rise_results)]:
        print(
            f"{name:<12} "
            f"{results['accuracy']:>10.3f} "
            f"{results['precision']:>10.3f} "
            f"{results['recall']:>10.3f} "
            f"{results['f1']:>10.3f}"
        )
    print()


if __name__ == "__main__":
    main()
