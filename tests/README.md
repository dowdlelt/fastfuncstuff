# Testing Guide for fastfuncsim

## Overview

Comprehensive test suite for the fastfuncsim package covering:
- ARMA noise generation and covariance structures
- GLM core functionality (OLS fitting, statistics)  
- fMRI simulation pipeline
- ARMA-GLM integration
- Output formats (NIfTI, AFNI)

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_arma_core.py -v
pytest tests/test_glm_core.py -v
pytest tests/test_simulation_core.py -v
```

### Run specific test class or function
```bash
pytest tests/test_arma_core.py::TestARMANoiseGeneration -v
pytest tests/test_glm_core.py::TestBasicGLM::test_simple_glm_fit -v
```

### Run with coverage
```bash
pytest tests/ --cov=fastfuncsim --cov-report=html
```

### Skip slow tests
```bash
pytest tests/ -m "not slow"
```

### Run only GPU tests
```bash
pytest tests/ -m gpu
```

## Test Organization

### `test_arma_core.py`
Comprehensive tests for ARMA functionality:
- **TestARMANoiseGeneration**: AR(1), MA(1), ARMA(1,1) noise
- **TestARMA11Covariance**: Covariance matrix construction
- **TestARMAParameterValidation**: Edge cases and validation

Example tests:
- Autocorrelation matches theoretical values
- Covariance matrices are symmetric and positive definite
- Parameter boundaries and stationarity

### `test_glm_core.py`
Comprehensive tests for GLM fitting:
- **TestBasicGLM**: OLS fitting with known signals
- **TestGLMStatistics**: t-stats, F-stats, R² computation
- **TestGLMEdgeCases**: Numerical stability, collinearity
- **TestGLMBatchProcessing**: Large-scale processing

Example tests:
- Beta recovery from known signals
- Statistical tests are finite and reasonable
- Residuals orthogonal to design matrix

### `test_simulation_core.py`
Comprehensive tests for simulation pipeline:
- **TestDesignGeneration**: Block and event-related designs
- **TestHRFConvolution**: Convolution accuracy and properties
- **TestFullSimulation**: End-to-end simulation
- **TestSimulationEdgeCases**: Edge cases and reproducibility

Example tests:
- HRF convolution preserves amplitude and timing
- Simulations with ARMA noise have autocorrelation
- Reproducibility with fixed seeds

### `test_arma_glm.py`
Integration tests for ARMA-GLM pipeline:
- ARMA(1,1) covariance construction
- REML parameter estimation
- Prewhitening transformations
- Full ARMA-GLM fitting

### `test_glm_outputs.py`
Tests for output formats:
- NIfTI export with metadata
- AFNI compatibility
- Multi-volume organization

## Writing New Tests

### Basic Test Structure
```python
import pytest
import torch
from fastfuncsim.utils import get_device

class TestMyFeature:
    """Test my feature."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_basic_functionality(self, device):
        """Test basic case."""
        # Arrange
        input_data = torch.randn(100, 10, device=device)
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result.shape == (100, 5)
        assert torch.all(torch.isfinite(result))
```

### Test Patterns

#### 1. Known Signal Recovery
```python
def test_known_signal(self, device):
    """Test with known ground truth."""
    true_value = torch.tensor([1.0, 2.0], device=device)
    estimated = my_estimator(data)
    assert torch.allclose(estimated, true_value, atol=0.1)
```

#### 2. Statistical Properties
```python
def test_statistical_property(self, device):
    """Test statistical properties."""
    noise = generate_noise(n=10000, device=device)
    assert abs(noise.mean()) < 0.05  # Mean ≈ 0
    assert abs(noise.std() - 1.0) < 0.05  # Std ≈ 1
```

#### 3. Edge Cases
```python
def test_edge_case(self, device):
    """Test edge case."""
    # Zero input
    result = my_function(torch.zeros(10, device=device))
    assert torch.all(torch.isfinite(result))
    
    # Single element
    result = my_function(torch.ones(1, device=device))
    assert result.shape == (1,)
```

#### 4. Reproducibility
```python
def test_reproducibility(self, device):
    """Test reproducibility with fixed seed."""
    torch.manual_seed(42)
    result1 = my_random_function(device=device)
    
    torch.manual_seed(42)
    result2 = my_random_function(device=device)
    
    assert torch.allclose(result1, result2)
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_large_scale():
    """Slow test with large data."""
    pass

@pytest.mark.gpu
def test_gpu_specific():
    """Test requiring GPU."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test."""
    pass
```

## Coverage Goals

Target coverage for each module:
- **arma_glm.py**: 90%+ (core ARMA functionality)
- **glm_core.py**: 90%+ (GLM fitting)
- **simulation.py**: 85%+ (simulation pipeline)
- **noise.py**: 85%+ (noise generation)
- **hrf.py**: 80%+ (HRF functions)

Check coverage:
```bash
pytest tests/ --cov=fastfuncsim --cov-report=term-missing
```

## Continuous Integration

Tests should:
- Run on CPU (always available)
- Gracefully handle GPU (test with MPS/CUDA if available)
- Complete in < 5 minutes for full suite
- Be deterministic (use fixed seeds)

## Debugging Failed Tests

### View detailed output
```bash
pytest tests/test_arma_core.py::test_name -v -s
```

### Stop at first failure
```bash
pytest tests/ -x
```

### Enter debugger on failure
```bash
pytest tests/ --pdb
```

### Show local variables
```bash
pytest tests/ -l
```

## Best Practices

1. **Test one thing per test**: Each test should verify a single aspect
2. **Use descriptive names**: Test names should explain what they test
3. **Keep tests fast**: Aim for < 1s per test when possible
4. **Use fixtures**: Share setup code with pytest fixtures
5. **Test edge cases**: Zero, negative, very large/small values
6. **Test error handling**: Ensure invalid inputs are caught
7. **Document expected behavior**: Use docstrings to explain tests
8. **Check finite values**: Always verify no NaN/Inf in results

## Adding Tests for New Features

When adding a new feature:

1. **Write tests first** (TDD approach)
2. **Test happy path** (basic functionality)
3. **Test edge cases** (boundaries, special values)
4. **Test error cases** (invalid inputs)
5. **Test integration** (how it works with existing code)
6. **Document expected behavior** in docstrings

## Resources

- pytest documentation: https://docs.pytest.org/
- PyTorch testing best practices: https://pytorch.org/docs/stable/testing.html
- Coverage.py: https://coverage.readthedocs.io/
