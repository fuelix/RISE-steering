# RISE: Rotor-Invariant Shift Estimation

Implementation of **Geometric Rotor Interpretations of Multilingual Embedding Models** (ICLR 2026).

RISE learns semantic transformations on the unit hypersphere using Riemannian geometry and Householder rotors, enabling cross-language transfer of transformations like negation, conditionality, and politeness shifts.

## Installation

```bash
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from rise import RISE, RISEPrototype
import torch

# Using the high-level RISE interface
rise = RISE()

# Fit on paired embeddings (neutral -> transformed)
neutral_embeddings = torch.randn(100, 768)  # Your embeddings
neutral_embeddings = torch.nn.functional.normalize(neutral_embeddings, dim=1)
transformed_embeddings = torch.randn(100, 768)
transformed_embeddings = torch.nn.functional.normalize(transformed_embeddings, dim=1)

stats = rise.fit(
    neutral_embeddings=neutral_embeddings,
    transformed_embeddings=transformed_embeddings
)

# Transform a new embedding
test_embedding = torch.randn(768)
test_embedding = torch.nn.functional.normalize(test_embedding, dim=0)
result = rise.transform(test_embedding)

print(f"Alignment: {result.metadata['alignment_score']:.4f}")
```

## Core Algorithm

RISE operates on the unit hypersphere S^(d-1) using Riemannian geometry:

1. **Riemannian Log Map**: Compute tangent vectors from neutral to transformed embeddings
   ```
   log_n(v) = (theta / sin(theta)) * (v - cos(theta) * n)
   ```

2. **Householder Rotor**: Canonicalize tangent vectors to a reference direction
   ```
   R(n) = I - 2 * u * u^T  where u = normalize(n - e_1)
   ```

3. **Prototype Learning**: Average canonicalized tangent vectors
   ```
   p = (1/N) * sum(R(n_i) @ log_{n_i}(v_i))
   ```

4. **Prediction**: Transport prototype to new base point via inverse rotor
   ```
   v_pred = exp_n(R(n)^T @ p)
   ```

## Baseline Comparisons

RISE includes implementations of comparison methods:

```python
from rise.baselines import ParkMethod, CAAMethod, HPRMethod

# Park's Linear Representation (ICML 2024)
park = ParkMethod(alpha=0.4)
park.fit(neutral_embeddings, transformed_embeddings)
result = park.transform(test_embedding)

# Contrastive Activation Addition (Rimsky et al., 2023)
caa = CAAMethod(strength=2.0)
caa.fit(neutral_embeddings, transformed_embeddings)
result = caa.transform(test_embedding)

# Householder Pseudo-Rotation (ACL 2024)
hpr = HPRMethod(n_reflections=2)
hpr.fit(neutral_embeddings, transformed_embeddings)
result = hpr.transform(test_embedding)
```

## Evaluation Metrics

```python
from rise.evaluation import (
    compute_alignment_score,
    compute_cross_language_transfer,
    compute_centroid_similarity,
)

# Alignment score (cosine similarity)
score = compute_alignment_score(predictions, targets)

# Cross-language transfer matrix
transfer = compute_cross_language_transfer(
    method=rise,
    language_data={"en": (en_neutral, en_transformed), ...},
    languages=["en", "es", "ja", "ar"]
)
```

## Project Structure

```
rise/
├── core/
│   ├── riemannian.py   # Log/exp maps on hypersphere
│   ├── rotor.py        # Householder rotor computation
│   ├── prototype.py    # Prototype learning & transport
│   └── rise.py         # High-level RISE interface
├── baselines/
│   ├── park.py         # Park et al. (2024)
│   ├── caa.py          # Rimsky et al. (2023)
│   └── hpr.py          # Chai et al. (2024)
├── evaluation/
│   ├── metrics.py      # Alignment scores, transfer metrics
│   └── visualization.py # Plotting utilities
└── utils/
    ├── constants.py    # Numerical constants
    ├── types.py        # Type definitions
    ├── config.py       # Configuration dataclasses
    └── reproducibility.py  # Seeding, logging, checkpoints
```

## Data

Pre-computed embeddings for reproducing paper results are available on HuggingFace:

**[mfwta/RISE-ICLR-2026](https://huggingface.co/datasets/mfwta/RISE-ICLR-2026)**

- 7 languages: English, Spanish, Japanese, Arabic, Thai, Tamil, Zulu
- 3 transformations: negation, conditionality, politeness
- 1,000 sentence pairs per language/transformation
- 3,072-dimensional OpenAI `text-embedding-3-large` embeddings

### Download Data

```bash
# Using huggingface_hub
pip install huggingface_hub
huggingface-cli download mfwta/RISE-ICLR-2026 --repo-type dataset --local-dir data/paper_embeddings

# Or clone with git
git clone https://huggingface.co/datasets/mfwta/RISE-ICLR-2026 data/paper_embeddings
```

## Reproducing Paper Results

```bash
# 1. Install the package
pip install -e .

# 2. Download the data (see above)

# 3. Run the evaluation suite
python -m rise.experiments.run_evaluation \
    --data-dir data/paper_embeddings \
    --transformations negation conditionality politeness \
    --languages en es ja ar th ta zu \
    --output-dir results/

# 4. Generate figures
python -m rise.experiments.generate_figures \
    --results-dir results/ \
    --output-dir figures/
```

To also run cross-language transfer experiments, add `--cross-language`:

```bash
python -m rise.experiments.run_evaluation \
    --data-dir data/paper_embeddings \
    --transformations negation conditionality politeness \
    --languages en es ja ar th ta zu \
    --output-dir results/ \
    --cross-language
```

## Citation

```bibtex
@inproceedings{rise2026,
  title={Geometric Rotor Interpretations of Multilingual Embedding Models},
  author={...},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
