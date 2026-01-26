"""
Main RISE interface for semantic transformations.

This module provides the high-level API for the RISE (Rotor-Invariant Shift
Estimation) method, combining prototype learning with embedding extraction
to provide an end-to-end interface for learning and applying semantic
transformations.

Example:
    ```python
    from rise.core import RISE

    # Initialize with an embedding function
    rise = RISE(embedder=my_embedding_function)

    # Learn from sentence pairs
    training_pairs = [
        ("The sky is blue.", "The sky is not blue."),
        ("I like coffee.", "I don't like coffee."),
        # ... more pairs
    ]
    rise.fit(training_pairs)

    # Predict transformation for new sentence
    predicted = rise.transform("The weather is nice.")
    ```
"""

import logging
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from .prototype import RISEPrototype, RISEPrototypeResult
from ..utils.types import AlignmentScore, TransformPrediction

logger = logging.getLogger(__name__)


# Type alias for embedding functions
EmbeddingFunction = Callable[[str], torch.Tensor]
BatchEmbeddingFunction = Callable[[List[str]], torch.Tensor]


class RISE:
    """
    High-level interface for the RISE semantic transformation method.

    RISE learns geometric transformations from paired sentences (e.g., original
    and negated versions) and can apply these transformations to new sentences.
    The method operates on the unit hypersphere where sentence embeddings reside.

    Attributes:
        prototype: The underlying RISEPrototype instance.
        embedder: Function to convert sentences to embeddings.
    """

    def __init__(
        self,
        embedder: Optional[Union[EmbeddingFunction, BatchEmbeddingFunction]] = None,
        canonicalize: bool = True,
    ):
        """
        Initialize a RISE instance.

        Args:
            embedder: Function that converts sentences to embeddings. Should accept
                      either a single string or a list of strings and return tensors.
                      If None, you must provide embeddings directly to fit/transform.
            canonicalize: Whether to use canonicalization (recommended for robustness).
        """
        self.prototype = RISEPrototype()
        self.embedder = embedder
        self.canonicalize = canonicalize
        self._is_fitted = False

    def fit(
        self,
        training_pairs: Optional[List[Tuple[str, str]]] = None,
        neutral_embeddings: Optional[torch.Tensor] = None,
        transformed_embeddings: Optional[torch.Tensor] = None,
    ) -> RISEPrototypeResult:
        """
        Learn a RISE prototype from training data.

        You can provide either sentence pairs (requires embedder) or
        pre-computed embeddings.

        Args:
            training_pairs: List of (neutral, transformed) sentence pairs.
                           Requires embedder to be set.
            neutral_embeddings: Pre-computed neutral embeddings, shape [M, d].
            transformed_embeddings: Pre-computed transformed embeddings, shape [M, d].

        Returns:
            RISEPrototypeResult with learning statistics.

        Raises:
            ValueError: If neither pairs nor embeddings are provided, or if
                       pairs are provided without an embedder.
        """
        if training_pairs is not None:
            if self.embedder is None:
                raise ValueError(
                    "Training pairs provided but no embedder set. "
                    "Either provide embeddings directly or set embedder in __init__."
                )

            logger.info(f"Embedding {len(training_pairs)} training pairs...")
            neutral_texts = [pair[0] for pair in training_pairs]
            transformed_texts = [pair[1] for pair in training_pairs]

            neutral_embeddings = self._embed_batch(neutral_texts)
            transformed_embeddings = self._embed_batch(transformed_texts)

        if neutral_embeddings is None or transformed_embeddings is None:
            raise ValueError(
                "Must provide either training_pairs or both "
                "neutral_embeddings and transformed_embeddings."
            )

        result = self.prototype.learn(
            neutral_embeddings,
            transformed_embeddings,
            canonicalize=self.canonicalize,
        )
        self._is_fitted = True

        return result

    def transform(
        self,
        text: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> TransformPrediction:
        """
        Predict the transformed embedding for a new input.

        Args:
            text: Input sentence (requires embedder to be set).
            embedding: Pre-computed embedding, shape [d].

        Returns:
            TransformPrediction with the predicted transformed embedding.

        Raises:
            ValueError: If not fitted, or if text is provided without embedder.
        """
        if not self._is_fitted:
            raise ValueError("RISE not fitted. Call fit() first.")

        if text is not None:
            if self.embedder is None:
                raise ValueError(
                    "Text provided but no embedder set. "
                    "Either provide embedding directly or set embedder in __init__."
                )
            embedding = self._embed_single(text)

        if embedding is None:
            raise ValueError("Must provide either text or embedding.")

        return self.prototype.predict(embedding)

    def evaluate(
        self,
        test_pairs: Optional[List[Tuple[str, str]]] = None,
        neutral_embeddings: Optional[torch.Tensor] = None,
        transformed_embeddings: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ) -> AlignmentScore:
        """
        Evaluate RISE predictions against ground truth transformations.

        Computes the alignment score (mean cosine similarity between predicted
        and actual transformed embeddings).

        Args:
            test_pairs: List of (neutral, transformed) sentence pairs.
            neutral_embeddings: Pre-computed neutral embeddings, shape [M, d].
            transformed_embeddings: Pre-computed transformed embeddings, shape [M, d].
            return_per_sample: If True, include individual scores in result.

        Returns:
            AlignmentScore with mean, std, and optionally per-sample scores.
        """
        if not self._is_fitted:
            raise ValueError("RISE not fitted. Call fit() first.")

        if test_pairs is not None:
            if self.embedder is None:
                raise ValueError("Test pairs provided but no embedder set.")

            neutral_texts = [pair[0] for pair in test_pairs]
            transformed_texts = [pair[1] for pair in test_pairs]

            neutral_embeddings = self._embed_batch(neutral_texts)
            transformed_embeddings = self._embed_batch(transformed_texts)

        if neutral_embeddings is None or transformed_embeddings is None:
            raise ValueError(
                "Must provide either test_pairs or both "
                "neutral_embeddings and transformed_embeddings."
            )

        # Normalize
        neutral_embeddings = F.normalize(neutral_embeddings, dim=1)
        transformed_embeddings = F.normalize(transformed_embeddings, dim=1)

        scores = []
        for i in range(len(neutral_embeddings)):
            prediction = self.prototype.predict(neutral_embeddings[i])
            score = F.cosine_similarity(
                prediction.predicted_embedding.unsqueeze(0),
                transformed_embeddings[i].unsqueeze(0),
            ).item()
            scores.append(score)

        scores_tensor = torch.tensor(scores)
        mean_score = scores_tensor.mean().item()
        std_score = scores_tensor.std().item()

        return AlignmentScore(
            score=mean_score,
            std=std_score,
            num_samples=len(scores),
            per_sample_scores=scores if return_per_sample else None,
        )

    def save(self, path: Path) -> None:
        """Save the fitted RISE model to a file."""
        if not self._is_fitted:
            raise ValueError("RISE not fitted. Call fit() first.")
        self.prototype.save(path)

    def load(self, path: Path) -> None:
        """Load a fitted RISE model from a file."""
        self.prototype = RISEPrototype.load(path)
        self._is_fitted = True

    def _embed_single(self, text: str) -> torch.Tensor:
        """Embed a single sentence."""
        if hasattr(self.embedder, "__call__"):
            result = self.embedder(text)
            if result.dim() == 2:
                result = result.squeeze(0)
            return result
        raise TypeError("Embedder must be callable")

    def _embed_batch(self, texts: List[str]) -> torch.Tensor:
        """Embed a batch of sentences."""
        if hasattr(self.embedder, "__call__"):
            # Try batch embedding first
            try:
                result = self.embedder(texts)
                if result.dim() == 1:
                    result = result.unsqueeze(0)
                return result
            except TypeError:
                # Fall back to single-sentence embedding
                embeddings = [self._embed_single(text) for text in texts]
                return torch.stack(embeddings)
        raise TypeError("Embedder must be callable")

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def prototype_norm(self) -> Optional[float]:
        """Norm of the learned prototype, or None if not fitted."""
        if self.prototype.prototype is None:
            return None
        return torch.norm(self.prototype.prototype).item()
