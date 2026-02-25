"""
Mean Difference Vector (MDV) baseline.

MDV computes the average Euclidean difference between transformed and neutral
embeddings, then applies this shift to predict transformations for new inputs.

This is the simplest baseline and does not account for the curved geometry of
the unit hypersphere.
"""

import torch
import torch.nn.functional as F

from ..utils.types import TransformPrediction


class MDV:
    """
    Mean Difference Vector baseline for semantic transformations.

    Computes prototype = mean(transformed - neutral) and predicts by
    adding this prototype to new embeddings, then re-normalizing.

    Implements the same duck-typed interface as RISE:
        .fit(neutral_embeddings, transformed_embeddings)
        .transform(embedding=tensor) -> TransformPrediction
    """

    def __init__(self):
        self.prototype: torch.Tensor | None = None
        self._is_fitted = False

    def fit(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
    ) -> None:
        """Learn the mean difference vector from paired embeddings."""
        neutral_embeddings = F.normalize(neutral_embeddings, dim=1)
        transformed_embeddings = F.normalize(transformed_embeddings, dim=1)

        diffs = transformed_embeddings - neutral_embeddings
        self.prototype = diffs.mean(dim=0)
        self._is_fitted = True

    def transform(
        self,
        embedding: torch.Tensor,
        text: str | None = None,
    ) -> TransformPrediction:
        """Predict the transformed embedding by adding the mean difference vector."""
        if not self._is_fitted:
            raise ValueError("MDV not fitted. Call fit() first.")

        embedding = F.normalize(embedding, dim=0)
        predicted = F.normalize(embedding + self.prototype, dim=0)
        cosine_sim = F.cosine_similarity(embedding, predicted, dim=0).item()

        return TransformPrediction(
            predicted_embedding=predicted,
            source_embedding=embedding,
            cosine_similarity=cosine_sim,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
