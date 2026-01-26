"""
RISE prototype learning and transport.

This module implements the core RISE algorithm for learning semantic transformation
prototypes from sentence pairs and applying them to new sentences.

Algorithm Overview:
    1. Canonicalization: For each (neutral, transformed) pair, compute a rotor
       R(n) that maps the neutral embedding n to the reference direction e_1.

    2. Prototype Learning: Compute the canonicalized tangent vectors and average:
       p̂_T = (1/M) Σ R(n_i) log_{n_i}(v_i)

    3. Prediction: For a new neutral embedding n*, transport the prototype back:
       v̂ = exp_{n*}(R(n*)^T p̂_T)

The prototype captures the "average direction" of the semantic transformation
in a canonical coordinate frame, making it applicable to any new sentence.

References:
    - RISE Paper: "Geometric Rotor Interpretations of Multilingual Embedding Models"
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from .riemannian import riemannian_log, riemannian_exp, project_to_tangent_space
from .rotor import compute_householder_rotor, apply_rotor, apply_rotor_transpose, get_reference_direction
from ..utils.constants import ORTHOGONALITY_TOL, ORTHOGONALITY_TOL_FP16
from ..utils.types import RISEPrototypeResult, TransformPrediction

logger = logging.getLogger(__name__)


class RISEPrototype:
    """
    RISE prototype for semantic transformations.

    This class encapsulates the learned prototype vector and provides methods
    for predicting transformations on new embeddings.

    Attributes:
        prototype: The learned prototype vector in tangent space at e_1.
        canonicalized: Whether canonicalization was used during learning.
        dim: Embedding dimensionality.
    """

    def __init__(self):
        """Initialize an empty RISE prototype."""
        self.prototype: Optional[torch.Tensor] = None
        self.canonicalized: bool = True
        self.dim: Optional[int] = None
        self._num_training_pairs: int = 0

    def learn(
        self,
        neutral_embeddings: torch.Tensor,
        transformed_embeddings: torch.Tensor,
        canonicalize: bool = True,
    ) -> RISEPrototypeResult:
        """
        Learn a RISE prototype from paired embeddings.

        Args:
            neutral_embeddings: Neutral sentence embeddings, shape [M, d]
            transformed_embeddings: Transformed sentence embeddings, shape [M, d]
            canonicalize: If True, use canonicalization (recommended). If False,
                          average tangent vectors directly (less robust).

        Returns:
            RISEPrototypeResult with the learned prototype and metadata.

        Raises:
            ValueError: If input tensors have mismatched shapes.
        """
        if neutral_embeddings.shape != transformed_embeddings.shape:
            raise ValueError(
                f"Shape mismatch: neutral {neutral_embeddings.shape} vs "
                f"transformed {transformed_embeddings.shape}"
            )

        M, d = neutral_embeddings.shape
        self.dim = d
        self.canonicalized = canonicalize

        logger.info(f"Learning RISE prototype from {M} pairs (canonicalize={canonicalize})")

        # Normalize all embeddings to unit sphere
        neutral_embeddings = F.normalize(neutral_embeddings, dim=1)
        transformed_embeddings = F.normalize(transformed_embeddings, dim=1)

        # Compute canonicalized tangent vectors
        tangent_vectors = []

        for i in range(M):
            n = neutral_embeddings[i]
            v = transformed_embeddings[i]

            # Step 1: Compute log map (tangent vector at n pointing to v)
            try:
                log_n_v = riemannian_log(n, v, verify_tangent_space=True)
            except ValueError as e:
                logger.warning(f"Skipping pair {i}: {e}")
                continue

            if canonicalize:
                # Step 2: Compute rotor that maps n to e_1
                R_n = compute_householder_rotor(n)

                # Step 3: Apply rotor to get canonicalized tangent
                # ξ = R(n) log_n(v)
                xi = apply_rotor(R_n, log_n_v)

                # Verify ξ is in tangent space at e_1
                e1 = get_reference_direction(d, device=n.device, dtype=n.dtype)
                tol = ORTHOGONALITY_TOL_FP16 if n.dtype == torch.float16 else ORTHOGONALITY_TOL
                dot_product = torch.abs(torch.dot(xi, e1))
                if dot_product >= tol:
                    logger.warning(
                        f"Pair {i}: canonicalized tangent not in T_{{e1}}: "
                        f"<ξ, e1> = {dot_product:.2e}"
                    )
            else:
                # Without canonicalization, use tangent vector directly
                xi = log_n_v

            tangent_vectors.append(xi)

        if not tangent_vectors:
            raise ValueError("No valid training pairs - all were skipped")

        # Step 4: Average tangent vectors to get prototype
        # p̂_T = (1/M) Σ ξ_i
        self.prototype = torch.stack(tangent_vectors).mean(dim=0)
        self._num_training_pairs = len(tangent_vectors)

        prototype_norm = torch.norm(self.prototype).item()
        logger.info(f"Prototype learned: ||p̂_T|| = {prototype_norm:.4f}")

        return RISEPrototypeResult(
            prototype=self.prototype,
            num_pairs=self._num_training_pairs,
            canonicalized=canonicalize,
            prototype_norm=prototype_norm,
            metadata={"skipped_pairs": M - len(tangent_vectors)},
        )

    def predict(
        self,
        neutral_embedding: torch.Tensor,
    ) -> TransformPrediction:
        """
        Predict the transformed embedding for a new neutral embedding.

        Uses the learned prototype to predict where the neutral embedding
        would move under the semantic transformation.

        Args:
            neutral_embedding: Neutral sentence embedding, shape [d]

        Returns:
            TransformPrediction with the predicted transformed embedding.

        Raises:
            ValueError: If prototype has not been learned.
        """
        if self.prototype is None:
            raise ValueError("Prototype not learned. Call learn() first.")

        # Ensure input is a unit vector
        n_star = F.normalize(neutral_embedding, dim=0)

        if self.canonicalized:
            # Transport prototype from T_{e1} to T_{n*}
            # Step 1: Compute rotor for target
            R_n_star = compute_householder_rotor(n_star)

            # Step 2: Apply inverse rotor to get tangent at n*
            # ξ* = R(n*)^T p̂_T
            prototype_local = self.prototype.to(device=R_n_star.device, dtype=R_n_star.dtype)
            xi_star = apply_rotor_transpose(R_n_star, prototype_local)
        else:
            # Without canonicalization, project prototype onto tangent space at n*
            prototype_local = self.prototype.to(device=n_star.device, dtype=n_star.dtype)
            xi_star = project_to_tangent_space(prototype_local, n_star)

        # Step 3: Apply exponential map to get predicted point
        # v̂ = exp_{n*}(ξ*)
        v_hat = riemannian_exp(n_star, xi_star, verify_tangent_space=True)

        cosine_sim = F.cosine_similarity(n_star, v_hat, dim=0).item()

        return TransformPrediction(
            predicted_embedding=v_hat,
            source_embedding=n_star,
            cosine_similarity=cosine_sim,
        )

    def save(self, path: Path) -> None:
        """
        Save the prototype to a file.

        Args:
            path: Path to save the prototype.
        """
        if self.prototype is None:
            raise ValueError("No prototype to save. Call learn() first.")

        state = {
            "prototype": self.prototype.cpu(),
            "canonicalized": self.canonicalized,
            "dim": self.dim,
            "num_training_pairs": self._num_training_pairs,
        }
        torch.save(state, path)
        logger.info(f"Prototype saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "RISEPrototype":
        """
        Load a prototype from a file.

        Args:
            path: Path to load the prototype from.

        Returns:
            Loaded RISEPrototype instance.
        """
        state = torch.load(path, weights_only=True)

        prototype = cls()
        prototype.prototype = state["prototype"]
        prototype.canonicalized = state["canonicalized"]
        prototype.dim = state["dim"]
        prototype._num_training_pairs = state["num_training_pairs"]

        logger.info(f"Prototype loaded from {path}")
        return prototype


def learn_rise_prototype(
    neutral_embeddings: torch.Tensor,
    transformed_embeddings: torch.Tensor,
    canonicalize: bool = True,
) -> torch.Tensor:
    """
    Convenience function to learn a RISE prototype.

    This is a functional interface for quick prototyping. For more control,
    use the RISEPrototype class directly.

    Args:
        neutral_embeddings: Neutral sentence embeddings, shape [M, d]
        transformed_embeddings: Transformed sentence embeddings, shape [M, d]
        canonicalize: Whether to use canonicalization (recommended).

    Returns:
        The learned prototype vector, shape [d]
    """
    rise = RISEPrototype()
    result = rise.learn(neutral_embeddings, transformed_embeddings, canonicalize)
    return result.prototype


def predict_transformation(
    prototype: torch.Tensor,
    neutral_embedding: torch.Tensor,
    canonicalized: bool = True,
) -> torch.Tensor:
    """
    Convenience function to predict a transformation.

    This is a functional interface for quick prototyping. For more control,
    use the RISEPrototype class directly.

    Args:
        prototype: The learned prototype vector, shape [d]
        neutral_embedding: Neutral sentence embedding, shape [d]
        canonicalized: Whether the prototype was learned with canonicalization.

    Returns:
        The predicted transformed embedding, shape [d]
    """
    rise = RISEPrototype()
    rise.prototype = prototype
    rise.canonicalized = canonicalized
    rise.dim = len(prototype)

    result = rise.predict(neutral_embedding)
    return result.predicted_embedding
