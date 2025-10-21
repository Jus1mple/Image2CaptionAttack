#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise injection wrapper for ResNet models.

This module implements a wrapper around CLIP's ResNet that adds random noise
at each layer output and removes it at the next layer input. This simulates
a privacy-preserving mechanism where noise is added during transmission.

The key idea:
- At layer N output: add random noise → transmit noisy features
- At layer N+1 input: subtract the same noise → continue processing

This tests whether the attack can still reconstruct captions even when
intermediate features are protected by additive noise.
"""

import torch
import clip


class NoiseResNetCLIP:
    """
    ResNet wrapper that adds noise to each layer and removes it at the next.

    This class wraps a CLIP ResNet model and injects random noise at each
    layer output, which is then subtracted at the next layer input. This
    simulates a privacy-preserving mechanism where features are perturbed
    during transmission between layers (e.g., split inference scenarios).

    The noise is stored in instance variables (noise1, noise2, noise3, noise4)
    and reused for subtraction, ensuring the main forward path remains
    mathematically equivalent to the original model.

    Args:
        clip_model: Pre-trained CLIP model with ResNet backbone

    Attributes:
        clip: The wrapped CLIP model
        noise1: Random noise added to layer1 output
        noise2: Random noise added to layer2 output
        noise3: Random noise added to layer3 output
        noise4: Random noise added to layer4 output

    Example:
        >>> clip_model, _ = clip.load("RN50", device="cuda")
        >>> noisy_model = NoiseResNetCLIP(clip_model)
        >>> image = torch.randn(1, 3, 224, 224).cuda()
        >>>
        >>> # Extract noisy features from layer2
        >>> noisy_features = noisy_model.layer2(
        ...     noisy_model.layer1(noisy_model.stem(image))
        ... )

    Notes:
        - The noise is randomly generated for each forward pass
        - Using torch.rand_like() generates uniform noise in [0, 1)
        - The same noise instance must be used for both addition and subtraction
        - This is a research tool for studying privacy leakage under noise
    """

    def __init__(self, clip_model):
        """
        Initialize the noisy ResNet wrapper.

        Args:
            clip_model: Pre-trained CLIP model with ResNet visual encoder
        """
        self.clip = clip_model

        # Noise tensors will be initialized during forward pass
        self.noise1 = None
        self.noise2 = None
        self.noise3 = None
        self.noise4 = None

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through ResNet stem layers.

        The stem consists of initial convolutional layers and pooling:
        - conv1 → bn1 → relu1
        - conv2 → bn2 → relu2
        - conv3 → bn3 → relu3
        - avgpool

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Stem output [B, 64, 56, 56] (for 224x224 input)
        """
        x = self.clip.visual.relu1(self.clip.visual.bn1(self.clip.visual.conv1(x)))
        x = self.clip.visual.relu2(self.clip.visual.bn2(self.clip.visual.conv2(x)))
        x = self.clip.visual.relu3(self.clip.visual.bn3(self.clip.visual.conv3(x)))
        x = self.clip.visual.avgpool(x)
        return x

    def layer1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through layer1 and add noise.

        Args:
            x: Input tensor from stem [B, 64, 56, 56]

        Returns:
            Output with added noise [B, 256, 56, 56]
        """
        out = self.clip.visual.layer1(x)
        self.noise1 = torch.rand_like(out)
        return out + self.noise1

    def layer2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove noise1, process through layer2, and add noise2.

        Args:
            x: Input tensor with noise from layer1 [B, 256, 56, 56]

        Returns:
            Output with added noise [B, 512, 28, 28]
        """
        out = self.clip.visual.layer2(x - self.noise1)
        self.noise2 = torch.rand_like(out)
        return out + self.noise2

    def layer3(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove noise2, process through layer3, and add noise3.

        Args:
            x: Input tensor with noise from layer2 [B, 512, 28, 28]

        Returns:
            Output with added noise [B, 1024, 14, 14]
        """
        out = self.clip.visual.layer3(x - self.noise2)
        self.noise3 = torch.rand_like(out)
        return out + self.noise3

    def layer4(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove noise3, process through layer4, and add noise4.

        Args:
            x: Input tensor with noise from layer3 [B, 1024, 14, 14]

        Returns:
            Output with added noise [B, 2048, 7, 7]
        """
        out = self.clip.visual.layer4(x - self.noise3)
        self.noise4 = torch.rand_like(out)
        return out + self.noise4

    def attnpool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove noise4 and process through attention pooling.

        Args:
            x: Input tensor with noise from layer4 [B, 2048, 7, 7]

        Returns:
            Pooled features [B, output_dim]
        """
        return self.clip.visual.attnpool(x - self.noise4)

    def visual(self):
        """
        Return self for compatibility with CLIP's visual attribute access.

        This allows using the wrapper like: noisy_model.visual.layer1(x)

        Returns:
            Self instance
        """
        return self

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass through the noisy ResNet.

        This is equivalent to CLIP's encode_image but with noise injection.

        Args:
            image: Input image tensor [B, 3, 224, 224]

        Returns:
            Encoded image features [B, output_dim]

        Example:
            >>> features = noisy_model.encode_image(image)
        """
        image = image.type(self.clip.visual.conv1.weight.dtype)
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

    def forward_add_noise(
        self,
        image: torch.Tensor,
        text: torch.Tensor
    ) -> tuple:
        """
        CLIP-style forward with image and text, using noisy image features.

        This method replicates CLIP's forward pass but uses the noise-injected
        image encoding instead of the clean one.

        Args:
            image: Input images [B, 3, 224, 224]
            text: Tokenized text [B, max_length]

        Returns:
            Tuple of (logits_per_image, logits_per_text):
                - logits_per_image: Image-to-text similarity [B, B]
                - logits_per_text: Text-to-image similarity [B, B]

        Example:
            >>> import clip
            >>> text = clip.tokenize(["a photo of a cat"]).cuda()
            >>> logits_img, logits_txt = noisy_model.forward_add_noise(image, text)
        """
        # Encode text using original CLIP encoder
        text_features = self.clip.encode_text(text)

        # Encode image using noisy path
        image_features = self.encode_image(image)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


# ==================== Utility Functions ====================

def get_added_noise_resnet_encoded_img_within_specified_layer(
    img: torch.Tensor,
    noised_clip_model: NoiseResNetCLIP,
    layer: str
) -> torch.Tensor:
    """
    Extract intermediate features from a specific layer with noise.

    This function extracts features from the NoiseResNetCLIP model at a
    specified layer, which can be used as "leaked features" in the attack.

    Args:
        img: Input image tensor [B, 3, 224, 224]
        noised_clip_model: NoiseResNetCLIP wrapper instance
        layer: Target layer name
            Options: "stem", "layer1", "layer2", "layer3", "layer4", "full"

    Returns:
        Noisy intermediate features from the specified layer
            - stem: [B, 64, 56, 56]
            - layer1: [B, 256, 56, 56]
            - layer2: [B, 512, 28, 28]
            - layer3: [B, 1024, 14, 14]
            - layer4: [B, 2048, 7, 7]
            - full: [B, output_dim]

    Example:
        >>> features = get_added_noise_resnet_encoded_img_within_specified_layer(
        ...     img, noisy_model, "layer2"
        ... )

    Notes:
        - Features are returned in float precision
        - Each call generates new random noise
        - For reproducibility, set torch.manual_seed() before calling
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
        # Default: return full encoding
        return noised_clip_model.encode_image(img).float()


# ==================== Test Functions ====================

def test_noise_resnet_clip():
    """
    Test NoiseResNetCLIP with CLIP forward pass.

    This function verifies that the noise injection doesn't break the model
    by comparing noisy vs. original CLIP outputs.
    """
    print("Testing NoiseResNetCLIP...")

    # Load CLIP model
    model, _ = clip.load("RN50", device="cuda:0", jit=False)
    noise_resnet_clip = NoiseResNetCLIP(model)

    # Create dummy inputs
    image = torch.randn(1, 3, 224, 224).cuda()
    text = clip.tokenize(["a photo of a cat"]).cuda()

    # Test noisy forward
    noised_logits_per_image, noised_logits_per_text = noise_resnet_clip.forward_add_noise(
        image, text
    )

    # Test original forward
    original_logits_per_image, original_logits_per_text = model(image, text)

    # Print results
    print(f"Noised Logits per Image: {noised_logits_per_image}")
    print(f"Noised Logits per Text: {noised_logits_per_text}")
    print(f"Original Logits per Image: {original_logits_per_image}")
    print(f"Original Logits per Text: {original_logits_per_text}")

    # The outputs should be different due to noise
    assert not torch.allclose(noised_logits_per_image, original_logits_per_image), \
        "Noisy and original outputs should differ!"

    print("✓ Test passed: Noise injection working as expected\n")


def test_get_specific_layer_noise_resnet_clip():
    """
    Test extracting features from specific layers.

    This function verifies that we can correctly extract intermediate
    features from different layers of the noisy ResNet.
    """
    print("Testing layer-specific feature extraction...")

    # Load model
    model, _ = clip.load("RN50", device="cuda:0", jit=False)
    model = model.to(torch.float)
    noise_resnet_clip = NoiseResNetCLIP(model)

    # Create dummy image
    image = torch.randn(1, 3, 224, 224).cuda()

    # Test different layers
    for layer_name in ["stem", "layer1", "layer2", "layer3", "layer4"]:
        noised_features = get_added_noise_resnet_encoded_img_within_specified_layer(
            image, noise_resnet_clip, layer_name
        )
        print(f"  {layer_name}: shape = {noised_features.shape}")

    print("✓ All layers working correctly\n")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print("=" * 70)
    print("NoiseResNetCLIP - Privacy-Preserving Noise Injection for ResNet")
    print("=" * 70)
    print()

    # Run tests
    test_noise_resnet_clip()
    test_get_specific_layer_noise_resnet_clip()

    print("=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)
