# RISE Test Suite

Comprehensive test suite for the RISE (Rotor-Invariant Shift Estimation) package.

## Overview

This test suite verifies the correctness, numerical stability, and robustness of RISE across:
- **Riemannian geometry operations** on the unit hypersphere
- **Householder rotor computation** for canonicalization
- **Prototype learning** from paired embeddings
- **High-level RISE interface** for semantic transformations

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures and test utilities
├── unit/                    # Unit tests for core modules
│   ├── test_riemannian.py  # Riemannian geometry operations
│   ├── test_rotor.py       # Householder rotor computation
│   ├── test_prototype.py   # Prototype learning and prediction
│   └── test_rise.py        # High-level RISE interface
└── numerical/               # Numerical stability tests
    └── test_stability.py   # Property-based tests with hypothesis
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=rise --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_riemannian.py -v
```

### Run specific test class
```bash
pytest tests/unit/test_riemannian.py::TestRiemannianLog -v
```

### Run specific test
```bash
pytest tests/unit/test_riemannian.py::TestRiemannianLog::test_log_returns_tangent_vector -v
```

### Run fast tests only (skip hypothesis)
```bash
pytest tests/unit/ -v
```

## Test Categories

### Unit Tests (`tests/unit/`)

#### `test_riemannian.py`
Tests for Riemannian geometry operations on S^{d-1}.

**Mathematical Properties Verified:**
- **Log-Exp Inverse**: `exp_n(log_n(v)) = v`
- **Exp-Log Inverse**: `log_n(exp_n(ξ)) = ξ` when ξ ⊥ n
- **Distance = Norm**: `geodesic_distance(n, v) = ||log_n(v)||`
- **Tangent Orthogonality**: `log_n(v) ⊥ n`

**Test Classes:**
- `TestRiemannianLog` - Logarithmic map tests
- `TestRiemannianExp` - Exponential map tests
- `TestLogExpInverse` - Inverse relationship tests
- `TestGeodesicDistance` - Distance computation tests
- `TestProjectToTangentSpace` - Tangent space projection tests
- `TestEdgeCases` - Boundary conditions and edge cases

#### `test_rotor.py`
Tests for Householder rotor computation.

**Mathematical Properties Verified:**
- **Rotor Orthogonality**: `R^T R = I`
- **Rotor Mapping**: `R @ source = target`
- **Rotor Determinant**: `det(R) = ±1`
- **Rotor Involution**: `R @ R = I` (reflection property)
- **Isometry**: Rotors preserve distances and angles

**Test Classes:**
- `TestComputeHouseholderRotor` - Rotor computation tests
- `TestApplyRotor` - Rotor application tests
- `TestVerifyOrthogonality` - Orthogonality verification tests
- `TestGetReferenceDirection` - Reference direction tests
- `TestRotorProperties` - Mathematical property tests
- `TestRotorEdgeCases` - Edge case handling

#### `test_prototype.py`
Tests for RISE prototype learning and prediction.

**Properties Verified:**
- **Prototype Unit Norm**: Predictions are on unit sphere
- **Learning Convergence**: Valid prototypes are learned from pairs
- **Persistence**: Save/load preserves prototype state
- **Determinism**: Same data produces same prototype
- **Consistency**: Multiple predictions are identical

**Test Classes:**
- `TestRISEPrototypeLearn` - Prototype learning tests
- `TestRISEPrototypePredict` - Prediction tests
- `TestRISEPrototypeSaveLoad` - Persistence tests
- `TestFunctionalAPI` - Convenience function tests
- `TestPrototypeConsistency` - Consistency and stability tests

#### `test_rise.py`
Tests for the high-level RISE interface.

**Test Classes:**
- `TestRISEInit` - Initialization tests
- `TestRISEFit` - Fitting tests (with text and embeddings)
- `TestRISETransform` - Transformation tests
- `TestRISEEvaluate` - Evaluation tests
- `TestRISESaveLoad` - Model persistence tests
- `TestRISEIntegration` - End-to-end workflow tests
- `TestRISEEmbedder` - Embedder handling tests

### Numerical Tests (`tests/numerical/`)

#### `test_stability.py`
Property-based tests using hypothesis for numerical stability.

**Features:**
- **Property-based testing**: Generates random inputs to verify properties hold universally
- **High-dimensional vectors**: Tests with dimensions up to 8192
- **Multiple dtypes**: float32, float16
- **Boundary conditions**: Near-zero, near-identity, near-antipodal cases
- **Stress testing**: Many training pairs, repeated predictions

**Test Classes:**
- `TestRiemannianProperties` - Riemannian geometry properties
- `TestRotorProperties` - Rotor mathematical properties
- `TestRISEPrototypeProperties` - Prototype learning properties
- `TestBoundaryConditions` - Edge cases and boundaries
- `TestStressConditions` - Stress tests

## Fixtures (conftest.py)

### Random Vector Generators
- `random_unit_vector(dim, dtype)` - Single random unit vector
- `random_unit_vector_batch(batch_size, dim, dtype)` - Batch of unit vectors
- `near_vectors(dim, angle_degrees, dtype)` - Nearly identical vectors
- `antipodal_vectors(dim, dtype)` - Nearly antipodal vectors

### Embedding Pairs
- `sample_embedding_pairs(num_pairs, dim, angular_shift, dtype)` - Synthetic embedding pairs for training

### Mock Embedders
- `mock_embedder(dim, dtype)` - Single-sentence embedder (deterministic)
- `mock_batch_embedder(dim, dtype)` - Batch embedder (deterministic)

### Parametrized Fixtures
- `test_dimension` - Tests across [64, 512, 3072] dimensions
- `test_dtype` - Tests across [float32, float16] dtypes

## Coverage Goals

Target coverage: **>80%** on core modules

Current coverage (after running tests):
```bash
pytest tests/ --cov=rise --cov-report=term-missing
```

Key modules:
- `rise.core.riemannian` - Target: 90%+
- `rise.core.rotor` - Target: 90%+
- `rise.core.prototype` - Target: 85%+
- `rise.core.rise` - Target: 85%+

## Performance

Test suite should complete in **<30 seconds** (unit tests) or **<2 minutes** (with hypothesis).

Benchmark:
```bash
time pytest tests/unit/ -v
```

For faster development iterations:
```bash
pytest tests/unit/ -v --maxfail=1  # Stop after first failure
```

## Mathematical Properties Reference

### Riemannian Geometry on S^{d-1}

The unit hypersphere S^{d-1} = {x ∈ ℝ^d : ||x|| = 1} with the Riemannian metric inherited from ℝ^d.

**Logarithmic Map:**
```
log_n(v) = (θ / sin(θ)) * (v - cos(θ) * n)
where θ = arccos(n · v)
```

**Exponential Map:**
```
exp_n(ξ) = cos(||ξ||) * n + sin(||ξ||) * (ξ / ||ξ||)
```

**Geodesic Distance:**
```
d(a, b) = arccos(a · b)
```

### Householder Reflections

A Householder reflection maps a vector x across a hyperplane:
```
H = I - 2 * v * v^T
```

Properties:
- H is orthogonal: H^T H = I
- H is involutory: H^2 = I
- H is a reflection: det(H) = -1

### RISE Algorithm

1. **Canonicalization**: Map neutral embeddings to reference direction e₁
   ```
   R(n) = compute_householder_rotor(n, e₁)
   ```

2. **Prototype Learning**: Average canonicalized tangent vectors
   ```
   p̂ = (1/M) Σ R(nᵢ) log_{nᵢ}(vᵢ)
   ```

3. **Prediction**: Transport prototype back to query tangent space
   ```
   v̂ = exp_{n*}(R(n*)^T p̂)
   ```

## Troubleshooting

### Common Test Failures

**"Vectors are nearly antipodal"**
- Expected behavior when testing boundary conditions
- Tests should handle this with pytest.raises()

**Numerical precision failures with float16**
- Use relaxed tolerances (1e-2 instead of 1e-5)
- Check ORTHOGONALITY_TOL_FP16 constant

**Hypothesis deadline exceeded**
- Add `@settings(deadline=None)` to slow tests
- Reduce `max_examples` for development

**Rotor orthogonality failures**
- Verify input vectors are actually unit norm
- Check for NaN/Inf values in inputs

### Running Tests in Debug Mode

```bash
pytest tests/unit/test_riemannian.py::TestRiemannianLog::test_log_returns_tangent_vector -v -s
```

The `-s` flag allows print statements to show in output.

### Skipping Slow Tests

```bash
pytest tests/ -v -m "not slow"
```

(Note: We haven't marked slow tests yet, but this is the pattern)

## Contributing

When adding new functionality to RISE:

1. **Write tests first** (TDD approach recommended)
2. **Cover edge cases** - especially numerical stability
3. **Document mathematical properties** being tested
4. **Use descriptive test names** - `test_what_is_being_tested_expected_behavior`
5. **Add docstrings** explaining what each test verifies
6. **Run full test suite** before committing

## CI/CD Integration

Recommended CI configuration:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest tests/ --cov=rise --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## License

MIT License - same as RISE package.
