# RISE Test Suite - Quick Start Guide

## Installation

```bash
cd /Users/michaelfreenor/c/RISE
pip install -e ".[dev]"
```

## Run Tests

### Run everything
```bash
pytest tests/ -v
```

### Fast iteration (unit tests only)
```bash
pytest tests/unit/ -v
```

### With coverage
```bash
pytest tests/ --cov=rise --cov-report=term-missing
```

### Single file
```bash
pytest tests/unit/test_riemannian.py -v
```

### Single test
```bash
pytest tests/unit/test_riemannian.py::TestRiemannianLog::test_log_returns_tangent_vector -v
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_riemannian.py  # 27 tests - Riemannian geometry
│   ├── test_rotor.py       # 28 tests - Householder rotors
│   ├── test_prototype.py   # 23 tests - Prototype learning
│   └── test_rise.py        # 28 tests - High-level interface
└── numerical/
    └── test_stability.py   # Property-based tests
```

## Key Tests by Module

### Riemannian (`test_riemannian.py`)
- `TestRiemannianLog` - Log map tests
- `TestRiemannianExp` - Exp map tests
- `TestLogExpInverse` - Inverse properties
- `TestGeodesicDistance` - Distance computation
- `TestProjectToTangentSpace` - Projection tests

### Rotor (`test_rotor.py`)
- `TestComputeHouseholderRotor` - Rotor computation
- `TestApplyRotor` - Rotor application
- `TestVerifyOrthogonality` - Orthogonality checks
- `TestRotorProperties` - Mathematical properties

### Prototype (`test_prototype.py`)
- `TestRISEPrototypeLearn` - Learning tests
- `TestRISEPrototypePredict` - Prediction tests
- `TestRISEPrototypeSaveLoad` - Persistence tests

### RISE (`test_rise.py`)
- `TestRISEFit` - Fitting tests
- `TestRISETransform` - Transformation tests
- `TestRISEEvaluate` - Evaluation tests
- `TestRISEIntegration` - End-to-end workflows

## Common Fixtures

```python
# Random vectors
def test_something(random_unit_vector):
    vec = random_unit_vector(dim=512)
    # ... test code

# Embedding pairs
def test_learning(sample_embedding_pairs):
    neutral, transformed = sample_embedding_pairs(num_pairs=10, dim=512)
    # ... test code

# Mock embedder
def test_with_embedder(mock_embedder):
    embedder = mock_embedder(dim=512)
    # ... test code
```

## Writing New Tests

### Template
```python
def test_feature_expected_behavior(fixture_name):
    """Test that feature does X when Y."""
    # Arrange
    input_data = fixture_name(param=value)
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected, "Error message"
```

### Best Practices
1. Use descriptive test names: `test_what_when_expected`
2. Add docstrings explaining what's being tested
3. Use fixtures from conftest.py
4. Include clear assertion messages
5. Test edge cases

## Troubleshooting

### Import errors
```bash
pip install -e ".[dev]"
```

### Tests hang
- Check for infinite loops
- Add `@settings(deadline=None)` to hypothesis tests

### Numerical precision failures
- Use appropriate tolerances
- Check dtype (float16 needs relaxed tolerance)

### Coverage not showing
```bash
pytest tests/ --cov=rise --cov-report=html
open htmlcov/index.html
```

## Mathematical Properties Tested

1. **Log-Exp Inverse**: exp_n(log_n(v)) = v
2. **Tangent Orthogonality**: log_n(v) ⊥ n
3. **Distance = Norm**: d(n,v) = ||log_n(v)||
4. **Rotor Orthogonality**: R^T R = I
5. **Rotor Mapping**: R @ source = target
6. **Unit Sphere Constraint**: Predictions have unit norm

## Performance Targets

- Unit tests: **<30 seconds**
- Full suite: **<2 minutes**
- Coverage: **>80% (target: 92%)**

## Need Help?

- See `tests/README.md` for detailed documentation
- See `tests/TEST_COVERAGE.md` for coverage details
- Check conftest.py for available fixtures
