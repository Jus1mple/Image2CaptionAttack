#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model architectures for Image2CaptionAttack.

This package contains:
- QFormer+OPT model for caption reconstruction from leaked features
- NoiseResNet wrapper for privacy-preserving noise injection
- Feature mapping modules for different victim model architectures
"""

from .qformer_opt import QFormerOptModel, ModelTask, Blip2ForConditionalGenerationModelOutput
from .noise_resnet import NoiseResNetCLIP, get_added_noise_resnet_encoded_img_within_specified_layer
from .feature_mappers import (
    # ResNet feature mappers
    FCN_Layer1, FCN_Layer2, FCN_Layer3, FCN_Layer4,
    # MobileNetV2 feature mappers
    MobileNetV2_FCN_Layer1, MobileNetV2_FCN_LayerALL, MobileNetV2_FCN_LayerMid,
    # MobileNetV3-Large feature mappers
    MobileNetV3L_FCN_Layer1, MobileNetV3L_FCN_LayerALL, MobileNetV3L_FCN_LayerMid,
    # MobileNetV3-Small feature mappers
    MobileNetV3S_FCN_Layer1, MobileNetV3S_FCN_LayerALL, MobileNetV3S_FCN_LayerMid,
    # Generic feature mapper
    FeatureMapper,
    # Base classes
    ResidualBlock, BaseFeatureMapper,
)

__all__ = [
    # Main attack model
    'QFormerOptModel',
    'ModelTask',
    'Blip2ForConditionalGenerationModelOutput',

    # Noise injection
    'NoiseResNetCLIP',
    'get_added_noise_resnet_encoded_img_within_specified_layer',

    # ResNet feature mappers
    'FCN_Layer1',
    'FCN_Layer2',
    'FCN_Layer3',
    'FCN_Layer4',

    # MobileNetV2 feature mappers
    'MobileNetV2_FCN_Layer1',
    'MobileNetV2_FCN_LayerALL',
    'MobileNetV2_FCN_LayerMid',

    # MobileNetV3-Large feature mappers
    'MobileNetV3L_FCN_Layer1',
    'MobileNetV3L_FCN_LayerALL',
    'MobileNetV3L_FCN_LayerMid',

    # MobileNetV3-Small feature mappers
    'MobileNetV3S_FCN_Layer1',
    'MobileNetV3S_FCN_LayerALL',
    'MobileNetV3S_FCN_LayerMid',

    # Generic and base classes
    'FeatureMapper',
    'ResidualBlock',
    'BaseFeatureMapper',
]
