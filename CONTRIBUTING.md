# Contributing to Image2CaptionAttack

Thank you for your interest in contributing to Image2CaptionAttack! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect everyone to:

- ✅ Be respectful and considerate
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what's best for the community
- ✅ Show empathy toward other contributors

### Unacceptable Behavior

- ❌ Harassment or discriminatory language
- ❌ Trolling or insulting comments
- ❌ Publishing others' private information
- ❌ Any conduct that could be considered unprofessional

---

## 🚀 Getting Started

### Prerequisites

Before contributing, make sure you have:

- Python 3.8 or higher
- Git installed and configured
- A GitHub account
- CUDA-capable GPU (optional, for testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/Image2CaptionAttack.git
cd Image2CaptionAttack
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/Image2CaptionAttack.git
```

---

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically check your code before each commit.

### 4. Verify Installation

```bash
# Test imports
python -c "from models import QFormerOptModel; print('✓ Setup successful')"
```

---

## 🔨 Making Contributions

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug fixes** - Fix issues in existing code
- ✨ **New features** - Add new functionality
- 📚 **Documentation** - Improve or add documentation
- 🧪 **Tests** - Add or improve test coverage
- 🎨 **Code quality** - Refactoring, optimization
- 🌐 **Datasets** - Add support for new datasets
- 🤖 **Models** - Add support for new victim models

### Before You Start

1. **Check existing issues** - See if someone is already working on it
2. **Create an issue** - Discuss major changes before implementing
3. **Get feedback** - Wait for maintainer feedback on your proposal

### Branch Naming Convention

Use descriptive branch names:

```bash
feature/add-convnext-support
bugfix/fix-memory-leak
docs/update-installation-guide
refactor/optimize-feature-extraction
```

---

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some project-specific conventions:

#### 1. Code Formatting

```bash
# Format code with black
black src/

# Sort imports with isort
isort src/

# Check style with flake8
flake8 src/
```

#### 2. Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def extract_features(image: torch.Tensor, layer: str) -> torch.Tensor:
    """
    Extract intermediate features from a specific layer.

    Args:
        image: Input image tensor [B, 3, H, W]
        layer: Target layer name (e.g., "layer2", "layer3")

    Returns:
        Extracted features [B, C, H', W']

    Raises:
        ValueError: If layer name is invalid

    Example:
        >>> features = extract_features(img, "layer2")
        >>> print(features.shape)
        torch.Size([1, 512, 28, 28])
    """
    pass
```

#### 3. Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Tuple

def process_batch(
    images: torch.Tensor,
    captions: List[str],
    max_length: int = 50
) -> Dict[str, torch.Tensor]:
    """Process a batch of images and captions."""
    pass
```

#### 4. Naming Conventions

```python
# Classes: PascalCase
class FeatureExtractor:
    pass

# Functions and variables: snake_case
def extract_features():
    batch_size = 32
    learning_rate = 1e-4

# Constants: UPPER_CASE
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 32

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Project-Specific Guidelines

#### 1. Module Organization

```python
# Standard library imports
import os
import sys
from typing import List, Optional

# Third-party imports
import torch
import torch.nn as nn
import numpy as np

# Project imports
from models import QFormerOptModel
from utils import load_checkpoint
from core import train, evaluate
```

#### 2. Error Handling

Always provide informative error messages:

```python
# Bad
if model is None:
    raise ValueError("Invalid model")

# Good
if model is None:
    raise ValueError(
        f"Model initialization failed. Expected QFormerOptModel instance, "
        f"but got None. Please check your model configuration."
    )
```

#### 3. Logging

Use the project's logger instead of print:

```python
# Bad
print("Training started")

# Good
logger.info("Training started on device: %s", device)
logger.debug("Batch size: %d, Learning rate: %.2e", batch_size, lr)
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

Create tests for new features:

```python
# tests/test_feature_extractor.py
import pytest
import torch
from utils import get_encode_fn


class TestFeatureExtractor:
    """Tests for feature extraction functions."""

    def test_resnet_layer2_extraction(self):
        """Test ResNet layer2 feature extraction."""
        encode_fn = get_encode_fn("resnet-layer2")
        img = torch.randn(1, 3, 224, 224)

        # Mock clip_model
        class MockClipModel:
            class Visual:
                layer1 = lambda self, x: x
                layer2 = lambda self, x: x
            visual = Visual()

        features = encode_fn(img, MockClipModel())
        assert features.shape[0] == 1  # Batch size
        assert features.dtype == torch.float32

    def test_invalid_layer_raises_error(self):
        """Test that invalid layer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown leaked feature layer"):
            get_encode_fn("invalid-layer")
```

### Test Coverage Goals

- **Minimum**: 70% code coverage
- **Target**: 85% code coverage
- **Critical paths**: 100% coverage for core functionality

---

## 📚 Documentation

### Updating Documentation

When adding new features, update:

1. **Docstrings** - All public functions and classes
2. **README.md** - Usage examples for new features
3. **API docs** - If applicable
4. **MIGRATION_GUIDE.md** - If changing existing APIs

### Documentation Style

```python
# Good documentation includes:
# 1. Clear description
# 2. Parameter explanations
# 3. Return value description
# 4. Usage examples
# 5. Error conditions

def train_model(
    model: QFormerOptModel,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4
) -> Dict[str, List[float]]:
    """
    Train the QFormer+OPT attack model.

    This function trains the model to reconstruct captions from leaked
    intermediate features using cross-entropy loss and AdamW optimizer.

    Args:
        model: QFormerOptModel instance to train
        dataloader: DataLoader with (features, captions) batches
        num_epochs: Number of training epochs. Default: 10
        learning_rate: Learning rate for AdamW optimizer. Default: 1e-4

    Returns:
        Dictionary containing training history:
            - 'loss': List of average loss per epoch
            - 'perplexity': List of perplexity per epoch

    Raises:
        ValueError: If model is not in training mode
        RuntimeError: If CUDA out of memory

    Example:
        >>> model = QFormerOptModel(blip2, ...)
        >>> history = train_model(model, train_loader, num_epochs=5)
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")

    Note:
        The model will be automatically set to training mode and moved
        to the appropriate device based on configuration.
    """
    pass
```

---

## 🔄 Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation

### 3. Commit Your Changes

Follow conventional commit format:

```bash
# Format: <type>(<scope>): <description>

git commit -m "feat(models): add ConvNeXt victim model support"
git commit -m "fix(utils): resolve memory leak in feature extraction"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(core): add tests for training loop"
git commit -m "refactor(models): optimize QFormer forward pass"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to GitHub and create a Pull Request
2. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added unit tests
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### 6. Code Review Process

- Maintainers will review your PR
- Address feedback and make requested changes
- Push updates to the same branch
- Once approved, maintainers will merge

---

## 🎯 Contribution Tips

### Good First Issues

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Community help needed
- `documentation` - Documentation improvements

### Getting Help

- 💬 **GitHub Discussions** - Ask questions
- 🐛 **GitHub Issues** - Report bugs
- 📧 **Email** - Contact maintainers

### Commit Message Best Practices

```bash
# Good commits
git commit -m "feat(models): add MobileNetV3-Small support

- Implement MobileNetV3S_FCN_Layer1/ALL/Mid classes
- Add feature dimension calculations
- Update model_loader.py with v3-small path
- Add comprehensive docstrings and type hints

Closes #123"

# Bad commits
git commit -m "fixed stuff"
git commit -m "update"
git commit -m "changes"
```

---

## 📋 Review Checklist

Before submitting a PR, ensure:

### Code Quality
- [ ] Code follows PEP 8 and project conventions
- [ ] All functions have docstrings
- [ ] Type hints are included
- [ ] No commented-out code
- [ ] No debug print statements

### Testing
- [ ] New features have tests
- [ ] All tests pass locally
- [ ] Test coverage maintained or improved
- [ ] Edge cases tested

### Documentation
- [ ] README updated (if needed)
- [ ] Docstrings added/updated
- [ ] MIGRATION_GUIDE updated (if API changed)
- [ ] Examples provided for new features

### Git
- [ ] Commits are atomic and well-described
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] Conventional commit format used

---

## 🙏 Recognition

Contributors will be recognized in:

- **README.md** - Contributors section
- **CHANGELOG.md** - Release notes
- **GitHub** - Contributors graph

---

## 📞 Questions?

- Open a GitHub Discussion for general questions
- Create an issue for bug reports or feature requests
- Email maintainers for private matters

---

Thank you for contributing to Image2CaptionAttack! Your efforts help make this project better for everyone. 🚀
