#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction utilities for different vision models.

This module provides functions to extract intermediate layer features from
various victim models (CLIP ViT, CLIP ResNet, MobileNet) for the attack.
"""

import torch
from functools import partial
from typing import Callable


# ==================== ViT Feature Extraction ====================

def get_vit_encoded_img(img: torch.Tensor, clip_model) -> torch.Tensor:
    """
    Extract final encoded image features from CLIP ViT model.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        clip_model: CLIP model with ViT backbone

    Returns:
        Encoded image features [B, feature_dim]
    """
    return clip_model.encode_image(img).float()


def get_vit_encoded_img_without_proj(img: torch.Tensor, clip_model) -> torch.Tensor:
    """
    Extract ViT features before the final projection layer.

    This function extracts features from the ViT model without applying
    the final linear projection, providing intermediate representations.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        clip_model: CLIP model with ViT backbone

    Returns:
        Encoded image features without projection [B, hidden_dim]
    """
    vit = clip_model.visual

    # Initial patch embedding
    x = vit.conv1(img)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # Add class token and positional embeddings
    x = torch.cat(
        [
            vit.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )  # shape = [*, grid ** 2 + 1, width]
    x = x + vit.positional_embedding.to(x.dtype)
    x = vit.ln_pre(x)

    # Transformer blocks
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = vit.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # Only layer norm, no projection
    x = vit.ln_post(x[:, 0, :])

    return x


# ==================== ResNet Feature Extraction ====================

def get_resnet_encoded_img(img: torch.Tensor, clip_model) -> torch.Tensor:
    """
    Extract final encoded image features from CLIP ResNet model.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        clip_model: CLIP model with ResNet backbone

    Returns:
        Encoded image features [B, feature_dim]
    """
    return clip_model.encode_image(img).float()


def get_resnet_encoded_img_within_specified_layer(
    img: torch.Tensor,
    clip_model,
    layer: str
) -> torch.Tensor:
    """
    Extract intermediate layer features from CLIP ResNet model.

    This function extracts features from a specific layer of the ResNet
    backbone, simulating feature leakage at different network depths.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        clip_model: CLIP model with ResNet backbone
        layer: Target layer name
            Options: "stem", "layer1", "layer2", "layer3", "layer4"

    Returns:
        Intermediate features from the specified layer
            - stem: [B, 64, 56, 56]
            - layer1: [B, 256, 56, 56]
            - layer2: [B, 512, 28, 28]
            - layer3: [B, 1024, 14, 14]
            - layer4: [B, 2048, 7, 7]

    Example:
        >>> features = get_resnet_encoded_img_within_specified_layer(
        ...     img, clip_model, "layer2"
        ... )
    """
    visual = clip_model.visual

    def stem(x):
        """Process through initial stem layers."""
        x = visual.relu1(visual.bn1(visual.conv1(x)))
        x = visual.relu2(visual.bn2(visual.conv2(x)))
        x = visual.relu3(visual.bn3(visual.conv3(x)))
        x = visual.avgpool(x)
        return x

    # Extract features from specified layer
    if layer == "stem":
        return stem(img).float()
    elif layer == "layer1":
        return visual.layer1(stem(img)).float()
    elif layer == "layer2":
        return visual.layer2(visual.layer1(stem(img))).float()
    elif layer == "layer3":
        return visual.layer3(visual.layer2(visual.layer1(stem(img)))).float()
    elif layer == "layer4":
        return visual.layer4(
            visual.layer3(visual.layer2(visual.layer1(stem(img))))
        ).float()
    else:
        # Default: return final encoding
        return get_resnet_encoded_img(img, clip_model)


def get_added_noise_resnet_encoded_img_within_specified_layer(
    img: torch.Tensor,
    noised_clip_model,
    layer: str
) -> torch.Tensor:
    """
    Extract features from a noise-added ResNet model.

    This function works with NoiseResNetCLIP wrapper that adds and removes
    noise at each layer to simulate privacy-preserving mechanisms.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        noised_clip_model: NoiseResNetCLIP wrapper instance
        layer: Target layer name
            Options: "stem", "layer1", "layer2", "layer3", "layer4"

    Returns:
        Intermediate features with added noise

    See Also:
        models.noise_resnet.NoiseResNetCLIP
    """
    if layer == "stem":
        return noised_clip_model.stem(img).float()
    elif layer == "layer1":
        return noised_clip_model.layer1(noised_clip_model.stem(img)).float()
    elif layer == "layer2":
        return noised_clip_model.layer2(
            noised_clip_model.layer1(noised_clip_model.stem(img))
        ).float()
    elif layer == "layer3":
        return noised_clip_model.layer3(
            noised_clip_model.layer2(
                noised_clip_model.layer1(noised_clip_model.stem(img))
            )
        ).float()
    elif layer == "layer4":
        return noised_clip_model.layer4(
            noised_clip_model.layer3(
                noised_clip_model.layer2(
                    noised_clip_model.layer1(noised_clip_model.stem(img))
                )
            )
        ).float()
    else:
        return noised_clip_model.encode_image_with_added_noise(img).float()


# ==================== MobileNet Feature Extraction ====================

def mobilenet_process_feature_layers(net, x: torch.Tensor, end_idx: int) -> torch.Tensor:
    """
    Process input through MobileNet feature layers up to a specific index.

    Args:
        net: MobileNet model
        x: Input tensor
        end_idx: Index of the last layer to process

    Returns:
        Features after processing through layers [0, end_idx]
    """
    for i, layer in enumerate(net.features):
        x = layer(x)
        if i == end_idx:
            break
    return x


def get_mobilenet_encoded_img_within_specified_layer(
    img: torch.Tensor,
    victim_model,
    layer: str
) -> torch.Tensor:
    """
    Extract intermediate layer features from MobileNet models.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        victim_model: MobileNet model (v2 or v3)
        layer: Target layer name
            Options:
            - "base": Final classification output [B, 1000]
            - "all-blocks": After all feature blocks
            - "layer1": After first layer
            - "layer-mid": After middle layers

    Returns:
        Intermediate features from the specified layer

    Raises:
        ValueError: If layer name is not recognized

    Example:
        >>> features = get_mobilenet_encoded_img_within_specified_layer(
        ...     img, mobilenet_model, "layer-mid"
        ... )
    """
    if layer == "base":
        return victim_model(img).float()
    elif layer == "all-blocks":
        return victim_model.features(img).float()
    elif layer == "layer1":
        return victim_model.features[0](img).float()
    elif layer == "layer-mid":
        mid_idx = len(victim_model.features) // 2 + 1
        return mobilenet_process_feature_layers(victim_model, img, mid_idx).float()
    else:
        raise ValueError(f"Unknown layer '{layer}' for MobileNet feature extraction")


# ==================== Feature Extraction Factory ====================

def get_encode_fn(leaked_feature_layer: str, add_noise: bool = False) -> Callable:
    """
    Get the appropriate feature extraction function for a given layer.

    This factory function returns the correct feature extraction function
    based on the victim model type and the target leaked layer.

    Args:
        leaked_feature_layer: Name of the leaked feature layer
            ViT options: "vit-base", "vit-no-proj"
            ResNet options: "resnet-base", "resnet-layer1/2/3/4"
            MobileNet options: "mobilenet-base", "mobilenet-all-blocks",
                              "mobilenet-layer1", "mobilenet-layer-mid"
        add_noise: Whether to use noise-added ResNet (only for ResNet layers)

    Returns:
        Feature extraction function that takes (img, model) as arguments

    Raises:
        ValueError: If leaked_feature_layer is not recognized

    Example:
        >>> encode_fn = get_encode_fn("resnet-layer2", add_noise=False)
        >>> features = encode_fn(images, clip_model)
    """
    # ViT feature extraction
    if leaked_feature_layer == "vit-base":
        return get_vit_encoded_img
    elif leaked_feature_layer == "vit-no-proj":
        return get_vit_encoded_img_without_proj

    # ResNet feature extraction
    elif leaked_feature_layer == "resnet-base":
        return get_resnet_encoded_img
    elif leaked_feature_layer in ["resnet-layer1", "resnet-layer2", "resnet-layer3", "resnet-layer4"]:
        layer_name = leaked_feature_layer.split("-")[1]  # Extract "layer1", "layer2", etc.
        extraction_fn = (
            get_added_noise_resnet_encoded_img_within_specified_layer
            if add_noise
            else get_resnet_encoded_img_within_specified_layer
        )
        return partial(extraction_fn, layer=layer_name)

    # MobileNet feature extraction
    elif leaked_feature_layer == "mobilenet-base":
        return partial(get_mobilenet_encoded_img_within_specified_layer, layer="base")
    elif leaked_feature_layer == "mobilenet-all-blocks":
        return partial(get_mobilenet_encoded_img_within_specified_layer, layer="all-blocks")
    elif leaked_feature_layer == "mobilenet-layer1":
        return partial(get_mobilenet_encoded_img_within_specified_layer, layer="layer1")
    elif leaked_feature_layer == "mobilenet-layer-mid":
        return partial(get_mobilenet_encoded_img_within_specified_layer, layer="layer-mid")

    else:
        raise ValueError(
            f"Unknown leaked feature layer: {leaked_feature_layer}. "
            f"See documentation for supported layer names."
        )
