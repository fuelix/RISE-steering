# RISE: Rotor-Invariant Shift Estimation

Implementation of **Mapping Semantic & Syntactic Relationships With Geometric Rotation** (ICLR 2026).

RISE learns semantic transformations on the unit hypersphere using Riemannian geometry and Householder rotors, enabling cross-language transfer of transformations like negation, conditionality, and politeness shifts.

## Installation

```bash
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Data

Pre-computed embeddings for reproducing paper results are available on HuggingFace:

**[mfwta/RISE-ICLR-2026](https://huggingface.co/datasets/mfwta/RISE-ICLR-2026)**

- 7 languages: English, Spanish, Japanese, Arabic, Thai, Tamil, Zulu
- 3 transformations: negation, conditionality, politeness
- 1,000 sentence pairs per language/transformation
- 3,072-dimensional OpenAI `text-embedding-3-large` embeddings

```bash
# Using huggingface_hub
pip install huggingface_hub
huggingface-cli download mfwta/RISE-ICLR-2026 --repo-type dataset --local-dir data/paper_embeddings
```

You can also use Python:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='mfwta/RISE-ICLR-2026',
    repo_type='dataset',
    local_dir='data/paper_embeddings'
)
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

## Reproducing Paper Results

```bash
# 1. Install the package
pip install -e .

# 2. Download the data (see above)

# 3. Verify paper results
python scripts/verify_paper_results.py

# Expected output:
# negation        0.857 (expected 0.864)  PASS
# conditionality  0.828 (expected 0.832)  PASS
# politeness      0.805 (expected 0.809)  PASS
```

### Full Evaluation Suite

```bash
python -m rise.experiments.run_evaluation \
    --data-dir data/paper_embeddings \
    --transformations negation conditionality politeness \
    --languages en es ja ar th ta zu \
    --output-dir results/

# Generate paper figures
python -m rise.experiments.generate_figures \
    --results-dir results/ \
    --output-dir figures/
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
