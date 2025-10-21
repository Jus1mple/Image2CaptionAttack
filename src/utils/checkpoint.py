#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint management utilities for saving and loading models.

This module provides functions to save and load model checkpoints,
specifically handling the QFormer+OPT model components.
"""

import os
import torch
from typing import Optional


def save_checkpoint(model, save_dir: str) -> None:
    """
    Save model checkpoint to disk.

    This function saves the QFormer+OPT model components as separate
    state dictionaries. Only the trainable components are saved:
    - intermediate_projection (maps intermediate features to QFormer input)
    - qformer (BLIP-2 Q-Former for vision-language alignment)
    - language_projection (projects QFormer output to language model input)
    - extra_conv_module (optional convolutional module for feature mapping)

    Args:
        model: QFormerOptModel instance to save
        save_dir: Directory path where checkpoint files will be saved

    Raises:
        OSError: If directory creation fails

    Example:
        >>> from models.qformer_opt import QFormerOptModel
        >>> model = QFormerOptModel(...)
        >>> save_checkpoint(model, "./checkpoints/epoch_10")
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save model configuration
    torch.save(model.config, os.path.join(save_dir, "config.json"))

    # Save optional convolutional module (if exists)
    if model.extra_conv_module is not None:
        torch.save(
            model.extra_conv_module.state_dict(),
            os.path.join(save_dir, "extra_conv_module_state_dict.pth"),
        )

    # Save intermediate projection layer
    torch.save(
        model.intermediate_projection.state_dict(),
        os.path.join(save_dir, "intermediate_projection_state_dict.pth"),
    )

    # Save QFormer
    torch.save(
        model.qformer.state_dict(),
        os.path.join(save_dir, "qformer_state_dict.pth"),
    )

    # Save language projection layer
    torch.save(
        model.language_projection.state_dict(),
        os.path.join(save_dir, "language_projection_state_dict.pth"),
    )

    print(f"✓ Checkpoint saved to: {save_dir}")


def load_checkpoint(model_path: str, model, device: str = "cuda:0"):
    """
    Load model checkpoint from disk.

    This function loads saved model components from the checkpoint directory
    and restores them into the provided model instance.

    Args:
        model_path: Path to the checkpoint directory
        model: QFormerOptModel instance to load weights into
        device: Device to load the model on (e.g., "cuda:0", "cpu")

    Returns:
        Model with loaded weights on the specified device

    Raises:
        FileNotFoundError: If checkpoint files are not found
        RuntimeError: If state dict loading fails due to architecture mismatch

    Example:
        >>> from models.qformer_opt import QFormerOptModel
        >>> model = QFormerOptModel(...)
        >>> model = load_checkpoint("./checkpoints/epoch_10", model, "cuda:0")

    Notes:
        - The function handles backward compatibility for old checkpoint names
          (clip_projection → intermediate_projection)
        - Extra conv module is loaded only if the checkpoint contains it
    """
    # Load optional convolutional module
    extra_conv_path = os.path.join(model_path, "extra_conv_module_state_dict.pth")
    if os.path.exists(extra_conv_path):
        if model.extra_conv_module is not None:
            model.extra_conv_module.load_state_dict(
                torch.load(extra_conv_path, map_location=device)
            )
            print(f"✓ Loaded extra_conv_module from {extra_conv_path}")
        else:
            print("⚠ Warning: Checkpoint contains extra_conv_module but model doesn't have one")

    # Load intermediate projection (with backward compatibility)
    try:
        intermediate_proj_path = os.path.join(
            model_path, "intermediate_projection_state_dict.pth"
        )
        model.intermediate_projection.load_state_dict(
            torch.load(intermediate_proj_path, map_location=device)
        )
        print(f"✓ Loaded intermediate_projection from {intermediate_proj_path}")
    except FileNotFoundError:
        # Backward compatibility: try old naming
        clip_proj_path = os.path.join(model_path, "clip_projection_state_dict.pth")
        model.intermediate_projection.load_state_dict(
            torch.load(clip_proj_path, map_location=device)
        )
        print(f"✓ Loaded intermediate_projection (old name) from {clip_proj_path}")

    # Load QFormer
    qformer_path = os.path.join(model_path, "qformer_state_dict.pth")
    model.qformer.load_state_dict(
        torch.load(qformer_path, map_location=device)
    )
    print(f"✓ Loaded qformer from {qformer_path}")

    # Load language projection
    lang_proj_path = os.path.join(model_path, "language_projection_state_dict.pth")
    model.language_projection.load_state_dict(
        torch.load(lang_proj_path, map_location=device)
    )
    print(f"✓ Loaded language_projection from {lang_proj_path}")

    # Move model to target device
    model = model.to(device)
    print(f"✓ Model loaded and moved to {device}")

    return model


def checkpoint_exists(model_path: str) -> bool:
    """
    Check if a valid checkpoint exists at the given path.

    Args:
        model_path: Path to the checkpoint directory

    Returns:
        True if all required checkpoint files exist, False otherwise

    Example:
        >>> if checkpoint_exists("./checkpoints/epoch_10"):
        ...     model = load_checkpoint("./checkpoints/epoch_10", model)
    """
    required_files = [
        "qformer_state_dict.pth",
        "language_projection_state_dict.pth",
    ]

    # Check for either new or old naming for intermediate projection
    has_intermediate_proj = (
        os.path.exists(os.path.join(model_path, "intermediate_projection_state_dict.pth"))
        or os.path.exists(os.path.join(model_path, "clip_projection_state_dict.pth"))
    )

    # Check all required files exist
    all_exist = all(
        os.path.exists(os.path.join(model_path, f)) for f in required_files
    )

    return all_exist and has_intermediate_proj
