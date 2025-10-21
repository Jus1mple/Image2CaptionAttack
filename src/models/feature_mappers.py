#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature mapping modules for different victim model architectures.

This module provides convolutional neural networks that map intermediate
features from different victim models (ResNet, MobileNet) to a common
1024-dimensional embedding space for the attack model.

All feature mappers follow the pattern:
    Leaked Features [C, H, W] → Conv Layers → Flatten → FC → [1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ==================== Base Residual Blocks ====================

class ResidualBlock(nn.Module):
    """
    Basic residual block with optional downsampling.

    Architecture:
        Conv3x3(stride) → BN → ReLU → Conv3x3 → BN → (+shortcut) → ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (1 or 2)

    Example:
        >>> block = ResidualBlock(64, 128, stride=2)
        >>> x = torch.randn(4, 64, 56, 56)
        >>> out = block(x)  # Shape: [4, 128, 28, 28]
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (identity or 1x1 conv)
        self.shortcut = None
        if (in_channels != out_channels) or (stride != 1):
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)
        return out


# ==================== Base Feature Mapper ====================

class BaseFeatureMapper(nn.Module):
    """
    Base class for feature mapping modules.

    Provides common functionality for all feature mappers that convert
    spatial features [C, H, W] to a 1024-dimensional vector.

    Args:
        conv_configs: List of (in_channels, out_channels, stride) tuples
        final_spatial_size: Expected (H, W) after all convolutions
        output_dim: Output dimension (default: 1024)
    """

    def __init__(
        self,
        conv_configs: List[Tuple[int, int, int]],
        final_spatial_size: Tuple[int, int],
        output_dim: int = 1024
    ):
        super(BaseFeatureMapper, self).__init__()

        # Build convolutional layers
        layers = []
        for in_c, out_c, stride in conv_configs:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ])
        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened dimension
        final_channels = conv_configs[-1][1]
        final_h, final_w = final_spatial_size
        flattened_dim = final_channels * final_h * final_w

        # Fully connected layer
        self.fc = nn.Linear(flattened_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# ==================== ResNet Feature Mappers ====================

class FCN_Layer1(BaseFeatureMapper):
    """
    Feature mapper for ResNet layer1 output.

    Input: [256, 56, 56] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (256, 512, 2),    # → [512, 28, 28]
            (512, 1024, 2),   # → [1024, 14, 14]
            (1024, 1024, 2),  # → [1024, 7, 7]
        ]
        super().__init__(conv_configs, final_spatial_size=(7, 7))


class FCN_Layer2(BaseFeatureMapper):
    """
    Feature mapper for ResNet layer2 output.

    Input: [512, 28, 28] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (512, 1024, 2),   # → [1024, 14, 14]
            (1024, 2048, 2),  # → [2048, 7, 7]
            (2048, 1024, 1),  # → [1024, 7, 7]
        ]
        super().__init__(conv_configs, final_spatial_size=(7, 7))


class FCN_Layer3(BaseFeatureMapper):
    """
    Feature mapper for ResNet layer3 output.

    Input: [1024, 14, 14] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (1024, 2048, 2),  # → [2048, 7, 7]
            (2048, 1024, 1),  # → [1024, 7, 7]
        ]
        super().__init__(conv_configs, final_spatial_size=(7, 7))


class FCN_Layer4(BaseFeatureMapper):
    """
    Feature mapper for ResNet layer4 output.

    Input: [2048, 7, 7] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (2048, 1024, 1),  # → [1024, 7, 7]
        ]
        super().__init__(conv_configs, final_spatial_size=(7, 7))


# ==================== MobileNetV2 Feature Mappers ====================

class MobileNetV2_FCN_Layer1(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV2 first layer output.

    Input: [32, 112, 112] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (32, 64, 2),     # → [64, 56, 56]
            (64, 128, 2),    # → [128, 28, 28]
            (128, 256, 2),   # → [256, 14, 14]
        ]
        super().__init__(conv_configs, final_spatial_size=(14, 14))


class MobileNetV2_FCN_LayerALL(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV2 all blocks output.

    Input: [320, 7, 7] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (320, 640, 2),   # → [640, 4, 4]
        ]
        super().__init__(conv_configs, final_spatial_size=(4, 4))


class MobileNetV2_FCN_LayerMid(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV2 middle layers output.

    Input: [64, 14, 14] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (64, 128, 2),    # → [128, 7, 7]
            (128, 256, 2),   # → [256, 4, 4]
        ]
        super().__init__(conv_configs, final_spatial_size=(4, 4))


# ==================== MobileNetV3-Large Feature Mappers ====================

class MobileNetV3L_FCN_Layer1(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV3-Large first layer output.

    Input: [16, 112, 112] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (16, 32, 2),     # → [32, 56, 56]
            (32, 64, 2),     # → [64, 28, 28]
            (64, 128, 2),    # → [128, 14, 14]
        ]
        super().__init__(conv_configs, final_spatial_size=(14, 14))


class MobileNetV3L_FCN_LayerALL(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV3-Large all blocks output.

    Input: [160, 7, 7] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (160, 320, 2),   # → [320, 4, 4]
            (320, 640, 2),   # → [640, 2, 2]
        ]
        super().__init__(conv_configs, final_spatial_size=(2, 2))


class MobileNetV3L_FCN_LayerMid(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV3-Large middle layers output.

    Input: [80, 14, 14] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (80, 160, 2),    # → [160, 7, 7]
            (160, 320, 2),   # → [320, 4, 4]
            (320, 640, 2),   # → [640, 2, 2]
        ]
        super().__init__(conv_configs, final_spatial_size=(2, 2))


# ==================== MobileNetV3-Small Feature Mappers ====================

class MobileNetV3S_FCN_Layer1(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV3-Small first layer output.

    Input: [16, 112, 112] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (16, 32, 2),     # → [32, 56, 56]
            (32, 64, 2),     # → [64, 28, 28]
            (64, 128, 2),    # → [128, 14, 14]
        ]
        super().__init__(conv_configs, final_spatial_size=(14, 14))


class MobileNetV3S_FCN_LayerALL(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV3-Small all blocks output.

    Input: [96, 7, 7] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (96, 192, 2),    # → [192, 4, 4]
            (192, 384, 2),   # → [384, 2, 2]
        ]
        super().__init__(conv_configs, final_spatial_size=(2, 2))


class MobileNetV3S_FCN_LayerMid(BaseFeatureMapper):
    """
    Feature mapper for MobileNetV3-Small middle layers output.

    Input: [48, 14, 14] → Output: [1024]
    """

    def __init__(self):
        conv_configs = [
            (48, 96, 2),     # → [96, 7, 7]
            (96, 192, 2),    # → [192, 4, 4]
            (192, 384, 2),   # → [384, 2, 2]
        ]
        super().__init__(conv_configs, final_spatial_size=(2, 2))


# ==================== Legacy Classes for Backward Compatibility ====================

class BasicBlock(nn.Module):
    """
    Basic ResNet block (not currently used, kept for compatibility).

    This is a standard ResNet BasicBlock implementation.
    """

    def __init__(self, inchannel: int, outchannel: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck ResNet block (not currently used, kept for compatibility).

    This is a standard ResNet Bottleneck implementation with 1x1 → 3x3 → 1x1 structure.
    """

    def __init__(self, inchannel: int, outchannel: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 4), kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 4), outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block (legacy, use ResidualBlock instead).

    Kept for backward compatibility with older checkpoints.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ConvNetWithResiduals(nn.Module):
    """
    Convolutional network with residual connections (legacy).

    Kept for backward compatibility with older checkpoints.
    """

    def __init__(self, input_channels: int, output_channels: int = 1024):
        super(ConvNetWithResiduals, self).__init__()

        self.layer1 = ResidualConvBlock(input_channels, 512, kernel_size=3, stride=2, padding=1)
        self.layer2 = ResidualConvBlock(512, 1024, kernel_size=3, stride=2, padding=1)
        self.layer3 = ResidualConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1)

        self.final_conv = nn.Conv2d(1024, output_channels, kernel_size=1, stride=1, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x


class FeatureMapper(nn.Module):
    """
    Generic feature mapper with configurable residual blocks.

    This module uses residual blocks to downsample and map features
    to a target dimension.

    Args:
        height: Input height
        width: Input width
        in_channels: Number of input channels
        out_dim: Output dimension (default: 1024)

    Example:
        >>> mapper = FeatureMapper(height=7, width=7, in_channels=160, out_dim=1024)
        >>> x = torch.randn(8, 160, 7, 7)
        >>> out = mapper(x)  # Shape: [8, 1024]
    """

    def __init__(self, height: int, width: int, in_channels: int = 160, out_dim: int = 1024):
        super(FeatureMapper, self).__init__()

        # Three residual blocks with progressive downsampling
        block1 = ResidualBlock(in_channels, in_channels * 2, stride=2)
        block2 = ResidualBlock(in_channels * 2, in_channels * 2, stride=1)
        block3 = ResidualBlock(in_channels * 2, in_channels * 4, stride=2)

        self.blocks = nn.Sequential(block1, block2, block3)

        # Calculate final spatial dimensions after downsampling
        self.final_channels = in_channels * 4
        self.final_height = (height // 2) // 2  # Two stride=2 blocks
        self.final_width = (width // 2) // 2

        # Fully connected layer
        self.fc = nn.Linear(self.final_channels * self.final_height * self.final_width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== Utility Functions ====================

def build_residual_conv_model(input_channels: int, output_channels: int) -> ConvNetWithResiduals:
    """
    Build a ConvNetWithResiduals model (legacy function).

    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels

    Returns:
        ConvNetWithResiduals instance
    """
    return ConvNetWithResiduals(input_channels, output_channels)


# ==================== Test Functions ====================

def test_resnet_feature_mappers():
    """Test all ResNet feature mappers."""
    print("=" * 60)
    print("Testing ResNet Feature Mappers")
    print("=" * 60)

    tests = [
        ("FCN_Layer1", FCN_Layer1(), (1, 256, 56, 56)),
        ("FCN_Layer2", FCN_Layer2(), (1, 512, 28, 28)),
        ("FCN_Layer3", FCN_Layer3(), (1, 1024, 14, 14)),
        ("FCN_Layer4", FCN_Layer4(), (1, 2048, 7, 7)),
    ]

    for name, model, input_shape in tests:
        x = torch.randn(*input_shape)
        output = model(x)
        print(f"  {name:15s} {str(input_shape):25s} → {tuple(output.shape)}")
        assert output.shape == (1, 1024), f"{name} output shape mismatch!"

    print("✓ All ResNet mappers passed!\n")


def test_mobilenet_feature_mappers():
    """Test all MobileNet feature mappers."""
    print("=" * 60)
    print("Testing MobileNet Feature Mappers")
    print("=" * 60)

    tests = [
        # MobileNetV2
        ("MobileNetV2_Layer1", MobileNetV2_FCN_Layer1(), (1, 32, 112, 112)),
        ("MobileNetV2_LayerALL", MobileNetV2_FCN_LayerALL(), (1, 320, 7, 7)),
        ("MobileNetV2_LayerMid", MobileNetV2_FCN_LayerMid(), (1, 64, 14, 14)),

        # MobileNetV3-Large
        ("MobileNetV3L_Layer1", MobileNetV3L_FCN_Layer1(), (1, 16, 112, 112)),
        ("MobileNetV3L_LayerALL", MobileNetV3L_FCN_LayerALL(), (1, 160, 7, 7)),
        ("MobileNetV3L_LayerMid", MobileNetV3L_FCN_LayerMid(), (1, 80, 14, 14)),

        # MobileNetV3-Small
        ("MobileNetV3S_Layer1", MobileNetV3S_FCN_Layer1(), (1, 16, 112, 112)),
        ("MobileNetV3S_LayerALL", MobileNetV3S_FCN_LayerALL(), (1, 96, 7, 7)),
        ("MobileNetV3S_LayerMid", MobileNetV3S_FCN_LayerMid(), (1, 48, 14, 14)),
    ]

    for name, model, input_shape in tests:
        x = torch.randn(*input_shape)
        output = model(x)
        print(f"  {name:25s} {str(input_shape):25s} → {tuple(output.shape)}")
        assert output.shape == (1, 1024), f"{name} output shape mismatch!"

    print("✓ All MobileNet mappers passed!\n")


def test_feature_mapper():
    """Test generic FeatureMapper."""
    print("=" * 60)
    print("Testing Generic FeatureMapper")
    print("=" * 60)

    c, h, w = 80, 14, 14
    out_dim = 1024
    x = torch.randn(8, c, h, w)

    mapper = FeatureMapper(height=h, width=w, in_channels=c, out_dim=out_dim)
    y = mapper(x)

    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")
    assert y.shape == (8, out_dim), "FeatureMapper output shape mismatch!"

    print("✓ Generic FeatureMapper passed!\n")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Feature Mappers for Image2CaptionAttack")
    print("=" * 60 + "\n")

    # Run all tests
    test_resnet_feature_mappers()
    test_mobilenet_feature_mappers()
    test_feature_mapper()

    print("=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
