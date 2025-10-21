#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model loading utilities for victim models.

This module provides functions to load various victim models including CLIP
(ViT and ResNet variants) and MobileNet models, as well as BLIP-2 model paths.
"""

import os
import sys
import torch
import clip
from typing import Optional


def return_blip2_model_path(blip2_model_name: str) -> Optional[str]:
    """
    Get the path to a BLIP-2 pretrained model.

    This function returns the local path where BLIP-2 models are stored.
    Paths should be configured via environment variables or config.py.

    Args:
        blip2_model_name: Name of the BLIP-2 model variant
            Options: "blip2-opt-2.7b", "blip2-opt-6.7b", "blip2-opt-6.7b-coco"

    Returns:
        Path to the model directory, or None if model name is not recognized

    Example:
        >>> path = return_blip2_model_path("blip2-opt-2.7b")
        >>> print(path)
        '/path/to/models/Salesforce/blip2-opt-2.7b'
    """
    # TODO: Move these paths to environment variables or config
    model_paths = {
        "blip2-opt-2.7b": "/root/autodl-tmp/models/Salesforce/blip2-opt-2.7b",
        "blip2-opt-6.7b": "/root/autodl-tmp/models/Salesforce/blip2-opt-6.7b",
        "blip2-opt-6.7b-coco": "/root/autodl-tmp/models/Salesforce/blip2-opt-6.7b-coco",
    }

    return model_paths.get(blip2_model_name, None)


def load_victim_model(victim_model_name: str, victim_device: str):
    """
    Load a victim model for feature extraction.

    This function loads different types of vision models that will be used
    as victim models for the feature leakage attack. Supports CLIP models
    (ViT and ResNet) and MobileNet variants.

    Args:
        victim_model_name: Name of the victim model
            CLIP models: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", etc.
            MobileNet: "mobilenetv2", "mobilenetv3-large", "mobilenetv3-small"
        victim_device: Device to load the model on (e.g., "cuda:0", "cpu")

    Returns:
        Loaded model in evaluation mode

    Raises:
        ImportError: If required model implementation cannot be imported
        FileNotFoundError: If model checkpoint file is not found

    Example:
        >>> model = load_victim_model("ViT-B/32", "cuda:0")
        >>> model = load_victim_model("mobilenetv2", "cuda:0")
    """
    # Load CLIP models (ViT or ResNet variants)
    if victim_model_name.startswith("ViT") or victim_model_name.startswith("RN"):
        model, _ = clip.load(victim_model_name, device=victim_device)
        model = model.to(torch.float)
        return model

    # Load MobileNet models
    elif victim_model_name.startswith("mobilenet"):
        return _load_mobilenet_model(victim_model_name, victim_device)

    else:
        raise ValueError(
            f"Unsupported victim model: {victim_model_name}. "
            f"Supported models: CLIP (ViT-*, RN*), MobileNet (mobilenetv2, mobilenetv3-*)"
        )


def _load_mobilenet_model(model_name: str, device: str):
    """
    Load MobileNet model variants.

    Internal helper function to load MobileNetV2 or MobileNetV3 models.

    Args:
        model_name: MobileNet variant name
        device: Device to load the model on

    Returns:
        Loaded MobileNet model
    """
    if model_name.startswith("mobilenetv2"):
        return _load_mobilenetv2(device)
    elif model_name.startswith("mobilenetv3"):
        return _load_mobilenetv3(model_name, device)
    else:
        raise ValueError(f"Unknown MobileNet variant: {model_name}")


def _load_mobilenetv2(device: str):
    """
    Load MobileNetV2 model.

    Requires: https://github.com/d-li14/mobilenetv2.pytorch

    Args:
        device: Device to load the model on

    Returns:
        Loaded MobileNetV2 model
    """
    from config import MobileNetV2Config

    sys.path.append(MobileNetV2Config.ROOT_DIR)

    try:
        # Try to import from the mobilenetv2.pytorch repository
        from mobilenetv2.models.imagenet import mobilenetv2
    except ImportError:
        # Fallback path
        sys.path.append("/root/repos/mobilenet")
        from mobilenetv2.models.imagenet import mobilenetv2

    model = mobilenetv2()
    model.load_state_dict(
        torch.load(MobileNetV2Config.MODEL_PATH, map_location=device)
    )
    model = model.to(device)
    model.eval()

    return model


def _load_mobilenetv3(model_name: str, device: str):
    """
    Load MobileNetV3 model (Large or Small variant).

    Requires: https://github.com/d-li14/mobilenetv3.pytorch

    Args:
        model_name: "mobilenetv3-large" or "mobilenetv3-small"
        device: Device to load the model on

    Returns:
        Loaded MobileNetV3 model
    """
    from config import MobileNetV3Config

    sys.path.append(MobileNetV3Config.ROOT_DIR)

    try:
        # Try to import from the mobilenetv3.pytorch repository
        from mobilenetv3.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
    except ImportError:
        # Fallback path
        sys.path.append("/root/repos/mobilenet")
        from mobilenetv3.mobilenetv3 import mobilenetv3_large, mobilenetv3_small

    if model_name == "mobilenetv3-large":
        model = mobilenetv3_large()
        model.load_state_dict(
            torch.load(MobileNetV3Config.LARGE_MODEL_PATH, map_location=device)
        )
    elif model_name == "mobilenetv3-small":
        model = mobilenetv3_small()
        model.load_state_dict(
            torch.load(MobileNetV3Config.SMALL_MODEL_PATH, map_location=device)
        )
    else:
        raise ValueError(
            f"Unknown MobileNetV3 variant: {model_name}. "
            f"Options: 'mobilenetv3-large', 'mobilenetv3-small'"
        )

    model = model.to(device)
    model.eval()

    return model
