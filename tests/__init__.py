"""
RISE Test Suite

Comprehensive tests for the RISE (Rotor-Invariant Shift Estimation) package.

Test Organization:
- unit/: Unit tests for individual modules
  - test_riemannian.py: Riemannian geometry operations
  - test_rotor.py: Householder rotor computation
  - test_prototype.py: Prototype learning and prediction
  - test_rise.py: High-level RISE interface

- numerical/: Numerical stability and property-based tests
  - test_stability.py: Hypothesis-based property testing

- conftest.py: Shared pytest fixtures and test utilities
"""
