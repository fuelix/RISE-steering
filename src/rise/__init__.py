"""
RISE: Rotor-Invariant Shift Estimation for Semantic Transformations.

This package implements the RISE method for learning geometric transformations
from paired sentence embeddings on the unit hypersphere.

Main classes:
    - RISE: High-level interface for learning and applying transformations
    - RISEPrototype: Low-level prototype learning and transport

Example:
    ```python
    from rise import RISE

    # With a custom embedder
    rise = RISE(embedder=my_embedding_function)
    rise.fit(training_pairs)
    prediction = rise.transform("New sentence to transform")

    # With pre-computed embeddings
    rise = RISE()
    rise.fit(
        neutral_embeddings=neutral_embeds,
        transformed_embeddings=transformed_embeds
    )
    ```
"""

from .core.rise import RISE
from .core.prototype import RISEPrototype, learn_rise_prototype, predict_transformation
from .core.riemannian import riemannian_log, riemannian_exp, geodesic_distance
from .core.rotor import compute_householder_rotor

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "RISE",
    "RISEPrototype",
    # Functional interface
    "learn_rise_prototype",
    "predict_transformation",
    # Riemannian geometry
    "riemannian_log",
    "riemannian_exp",
    "geodesic_distance",
    # Rotor computation
    "compute_householder_rotor",
]
