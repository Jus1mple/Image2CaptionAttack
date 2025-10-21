#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random seed utilities for reproducibility.

This module provides functions to set random seeds across different libraries
(Python random, NumPy, PyTorch) to ensure reproducible results.
"""

import random
import numpy as np
import torch


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    This function sets the random seed for Python's random module, NumPy,
    and PyTorch (both CPU and CUDA). It also sets CuDNN to deterministic mode.

    Args:
        seed: Random seed value. Default: 42

    Example:
        >>> set_random_seeds(42)
        >>> # Now all random operations will be reproducible
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (single GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # PyTorch CUDA (all GPUs)
        torch.cuda.manual_seed_all(seed)

    # CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test the function
    set_random_seeds(42)
    print(f"Random int: {random.randint(0, 100)}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
