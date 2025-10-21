#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core functionality for Image2CaptionAttack.

This package contains:
- Training loop implementation
- Evaluation/prediction functions
- Data collation utilities
"""

from .trainer import train
from .predictor import evaluate
from .data_collator import collate_fn, collate_fn_for_ocr

__all__ = [
    'train',
    'evaluate',
    'collate_fn',
    'collate_fn_for_ocr',
]
