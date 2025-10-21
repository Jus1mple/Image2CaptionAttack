#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset factory utilities for creating dataset processors.

This module provides factory functions to instantiate the correct dataset
processor based on dataset name, and to initialize feature mapping modules.
"""

from typing import Optional, Tuple, Union, List
import torch.nn as nn


def get_dataset_processor(dataset_name: str):
    """
    Get the appropriate dataset processor class for a given dataset.

    This factory function returns the correct dataset processor class
    that can load and process images and captions from different datasets.

    Args:
        dataset_name: Name of the dataset
            Options: "COCO2017", "flickr8k", "imagenet"

    Returns:
        Dataset processor class (not instantiated)

    Raises:
        ValueError: If dataset_name is not recognized

    Example:
        >>> ProcessorClass = get_dataset_processor("COCO2017")
        >>> processor = ProcessorClass(
        ...     data_dir="/path/to/images",
        ...     annotation_file="/path/to/annotations.json",
        ...     caption_file="/path/to/captions.json"
        ... )

    Notes:
        - The returned value is a class, not an instance
        - You need to instantiate it with appropriate arguments
        - Each dataset has different required arguments
    """
    if dataset_name == "flickr8k":
        from dataset import Flickr8kProcessor
        return Flickr8kProcessor

    elif dataset_name == "COCO2017":
        from dataset import COCOProcessor
        return COCOProcessor

    elif dataset_name == "imagenet":
        from dataset import ImageNetCapProcessor
        return ImageNetCapProcessor

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: 'COCO2017', 'flickr8k', 'imagenet'"
        )


def init_intermediate_dim_and_extra_conv_module(
    victim_model_name: str,
    leaked_feature_layer: str
) -> Tuple[Union[int, List[int]], Optional[nn.Module]]:
    """
    Initialize intermediate feature dimension and extra convolutional module.

    This function determines the feature dimension of the leaked intermediate
    layer and creates an appropriate convolutional module to map those features
    to a common embedding space (typically 1024-dim).

    Args:
        victim_model_name: Name of the victim model
            Options: "ViT-*", "RN*", "mobilenetv2", "mobilenetv3-large", "mobilenetv3-small"
        leaked_feature_layer: Name of the leaked layer
            ViT: "vit-base", "vit-no-proj"
            ResNet: "resnet-base", "resnet-layer1/2/3/4"
            MobileNet: "mobilenet-base", "mobilenet-layer1", "mobilenet-all-blocks", "mobilenet-layer-mid"

    Returns:
        Tuple of (intermediate_feature_dim, extra_conv_module):
            - intermediate_feature_dim: Feature dimension (int) or [C, H, W] (list)
            - extra_conv_module: Convolutional module to map features (or None)

    Raises:
        ValueError: If layer name is not recognized
        AssertionError: If MobileNet layer dimension is not a list

    Example:
        >>> dim, conv_module = init_intermediate_dim_and_extra_conv_module(
        ...     "RN50", "resnet-layer2"
        ... )
        >>> print(dim)  # Output: 1024
        >>> print(conv_module)  # Output: FCN_Layer2()
    """
    from config import return_compressed_image_feature_dim
    from residual_conv_model import (
        FCN_Layer1, FCN_Layer2, FCN_Layer3, FCN_Layer4,
        MobileNetV2_FCN_Layer1, MobileNetV2_FCN_LayerALL, MobileNetV2_FCN_LayerMid,
        MobileNetV3L_FCN_Layer1, MobileNetV3L_FCN_LayerALL, MobileNetV3L_FCN_LayerMid,
        MobileNetV3S_FCN_Layer1, MobileNetV3S_FCN_LayerALL, MobileNetV3S_FCN_LayerMid,
    )

    # Get intermediate feature dimension from config
    intermediate_feature_dim = return_compressed_image_feature_dim(
        victim_model_name, leaked_feature_layer
    )

    # ==================== ViT Models ====================
    if victim_model_name.startswith("ViT"):
        # ViT doesn't need extra conv module
        extra_conv_module = None

    # ==================== ResNet Models ====================
    elif leaked_feature_layer == "resnet-base":
        # Final ResNet features don't need extra conv module
        extra_conv_module = None

    elif leaked_feature_layer.startswith("resnet"):
        # Intermediate ResNet layers need conv module to map to 1024-dim
        intermediate_feature_dim = 1024

        if "layer1" in leaked_feature_layer:
            # Input: [256, 56, 56] → Output: [1024]
            extra_conv_module = FCN_Layer1()
        elif "layer2" in leaked_feature_layer:
            # Input: [512, 28, 28] → Output: [1024]
            extra_conv_module = FCN_Layer2()
        elif "layer3" in leaked_feature_layer:
            # Input: [1024, 14, 14] → Output: [1024]
            extra_conv_module = FCN_Layer3()
        elif "layer4" in leaked_feature_layer:
            # Input: [2048, 7, 7] → Output: [1024]
            extra_conv_module = FCN_Layer4()
        else:
            raise ValueError(
                f"Unknown ResNet layer: {leaked_feature_layer}. "
                f"Options: resnet-base, resnet-layer1/2/3/4"
            )

    # ==================== MobileNet Models ====================
    elif leaked_feature_layer.startswith("mobilenet"):

        if leaked_feature_layer == "mobilenet-base":
            # Final classification output (1000-dim)
            intermediate_feature_dim = 1000
            extra_conv_module = None
        else:
            # Intermediate layers return [C, H, W] shape
            assert isinstance(intermediate_feature_dim, list), (
                "Error! The intermediate_feature_dim should be a list [C, H, W] "
                f"for MobileNet layers, but got {type(intermediate_feature_dim)}"
            )

            # Map to 1024-dim using appropriate FCN module
            intermediate_feature_dim = 1024

            # Select appropriate FCN based on model variant and layer
            if victim_model_name == "mobilenetv2":
                extra_conv_module = _get_mobilenetv2_conv_module(leaked_feature_layer)
            elif victim_model_name == "mobilenetv3-large":
                extra_conv_module = _get_mobilenetv3_large_conv_module(leaked_feature_layer)
            elif victim_model_name == "mobilenetv3-small":
                extra_conv_module = _get_mobilenetv3_small_conv_module(leaked_feature_layer)
            else:
                raise ValueError(f"Unknown MobileNet variant: {victim_model_name}")

    else:
        raise ValueError(
            f"Unknown model or layer combination: {victim_model_name}, {leaked_feature_layer}"
        )

    return intermediate_feature_dim, extra_conv_module


def _get_mobilenetv2_conv_module(leaked_feature_layer: str) -> nn.Module:
    """
    Get appropriate FCN module for MobileNetV2 layers.

    Args:
        leaked_feature_layer: Layer name

    Returns:
        Corresponding FCN module

    Raises:
        ValueError: If layer name is not recognized
    """
    from residual_conv_model import (
        MobileNetV2_FCN_Layer1,
        MobileNetV2_FCN_LayerALL,
        MobileNetV2_FCN_LayerMid,
    )

    if leaked_feature_layer == "mobilenet-layer1":
        # Input: [32, 112, 112] → Output: [1024]
        return MobileNetV2_FCN_Layer1()
    elif leaked_feature_layer == "mobilenet-all-blocks":
        # Input: [320, 7, 7] → Output: [1024]
        return MobileNetV2_FCN_LayerALL()
    elif leaked_feature_layer == "mobilenet-layer-mid":
        # Input: [64, 14, 14] → Output: [1024]
        return MobileNetV2_FCN_LayerMid()
    else:
        raise ValueError(
            f"Unknown MobileNetV2 layer: {leaked_feature_layer}. "
            f"Options: mobilenet-layer1, mobilenet-all-blocks, mobilenet-layer-mid"
        )


def _get_mobilenetv3_large_conv_module(leaked_feature_layer: str) -> nn.Module:
    """
    Get appropriate FCN module for MobileNetV3-Large layers.

    Args:
        leaked_feature_layer: Layer name

    Returns:
        Corresponding FCN module

    Raises:
        ValueError: If layer name is not recognized
    """
    from residual_conv_model import (
        MobileNetV3L_FCN_Layer1,
        MobileNetV3L_FCN_LayerALL,
        MobileNetV3L_FCN_LayerMid,
    )

    if leaked_feature_layer == "mobilenet-layer1":
        # Input: [16, 112, 112] → Output: [1024]
        return MobileNetV3L_FCN_Layer1()
    elif leaked_feature_layer == "mobilenet-all-blocks":
        # Input: [160, 7, 7] → Output: [1024]
        return MobileNetV3L_FCN_LayerALL()
    elif leaked_feature_layer == "mobilenet-layer-mid":
        # Input: [80, 14, 14] → Output: [1024]
        return MobileNetV3L_FCN_LayerMid()
    else:
        raise ValueError(
            f"Unknown MobileNetV3-Large layer: {leaked_feature_layer}. "
            f"Options: mobilenet-layer1, mobilenet-all-blocks, mobilenet-layer-mid"
        )


def _get_mobilenetv3_small_conv_module(leaked_feature_layer: str) -> nn.Module:
    """
    Get appropriate FCN module for MobileNetV3-Small layers.

    Args:
        leaked_feature_layer: Layer name

    Returns:
        Corresponding FCN module

    Raises:
        ValueError: If layer name is not recognized
    """
    from residual_conv_model import (
        MobileNetV3S_FCN_Layer1,
        MobileNetV3S_FCN_LayerALL,
        MobileNetV3S_FCN_LayerMid,
    )

    if leaked_feature_layer == "mobilenet-layer1":
        # Input: [16, 112, 112] → Output: [1024]
        return MobileNetV3S_FCN_Layer1()
    elif leaked_feature_layer == "mobilenet-all-blocks":
        # Input: [96, 7, 7] → Output: [1024]
        return MobileNetV3S_FCN_LayerALL()
    elif leaked_feature_layer == "mobilenet-layer-mid":
        # Input: [48, 14, 14] → Output: [1024]
        return MobileNetV3S_FCN_LayerMid()
    else:
        raise ValueError(
            f"Unknown MobileNetV3-Small layer: {leaked_feature_layer}. "
            f"Options: mobilenet-layer1, mobilenet-all-blocks, mobilenet-layer-mid"
        )
