#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility modules for Image2CaptionAttack.

This package provides utilities for:
- Random seed management
- Model loading (CLIP, MobileNet, BLIP-2)
- Feature extraction from intermediate layers
- Checkpoint saving and loading
- Dataset processor factory
"""

from .random_seed import set_random_seeds
from .model_loader import load_victim_model, return_blip2_model_path
from .feature_extractor import get_encode_fn
from .checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from .dataset_factory import get_dataset_processor, init_intermediate_dim_and_extra_conv_module

__all__ = [
    # Random seed utilities
    'set_random_seeds',

    # Model loading
    'load_victim_model',
    'return_blip2_model_path',

    # Feature extraction
    'get_encode_fn',

    # Checkpoint management
    'save_checkpoint',
    'load_checkpoint',
    'checkpoint_exists',

    # Dataset factory
    'get_dataset_processor',
    'init_intermediate_dim_and_extra_conv_module',
]
