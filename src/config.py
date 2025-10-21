#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for Image2CaptionAttack project.

This module contains:
- Model configurations and supported architectures
- Command-line argument parsing
- Feature dimension calculations for different victim models
- Environment variable-based path configurations
"""

import os
import argparse


# ==================== Model Configuration ====================

# Main model architecture name
MODEL_NAME = "QformerOPTModel"

# Supported BLIP-2 model variants
BLIP_MODEL_NAMES = [
    "blip2-opt-2.7b",        # BLIP-2 with OPT-2.7B language model
    "blip2-opt-6.7b",        # BLIP-2 with OPT-6.7B language model
    "blip2-opt-6.7b-coco"    # BLIP-2 with OPT-6.7B fine-tuned on COCO
]

# Supported victim models (feature extractors)
VICTIM_MODEL_NAMES = {
    # ResNet models
    "RN50": "RN50",
    "RN101": "RN101",
    # "RN50x4": "RN50x4",          # Uncomment if needed
    # "RN50x16": "RN50x16",
    # "RN50x64": "RN50x64",

    # Vision Transformer models
    "ViT-32B": "ViT-B/32",
    "ViT-16B": "ViT-B/16",
    # "ViT-14L": "ViT-L/14",
    # "ViT-14L-336px": "ViT-L/14@336px",

    # MobileNet models
    "mobilenetv2": "mobilenetv2",
    "mobilenetv3-large": "mobilenetv3-large",
    "mobilenetv3-small": "mobilenetv3-small",
}

# Model running modes
_TRAIN = "train"   # Training mode
_TEST = "test"     # Evaluation mode
_WHOLE = "whole"   # Complete training + evaluation pipeline

MODEL_MODES = [_TRAIN, _TEST, _WHOLE]



# ==================== MobileNet Model Paths ====================

class MobileNetV2Config:
    """Configuration for MobileNetV2 pretrained models."""
    ROOT_DIR = os.getenv("MOBILENET_DIR", "./pretrained_models/mobilenet")
    MODEL_PATH = os.getenv(
        "MOBILENET_V2_PATH",
        os.path.join(ROOT_DIR, "mobilenetv2/pretrained/mobilenetv2-c5e733a8.pth")
    )


class MobileNetV3Config:
    """Configuration for MobileNetV3 pretrained models."""
    ROOT_DIR = os.getenv("MOBILENET_DIR", "./pretrained_models/mobilenet")
    LARGE_MODEL_PATH = os.getenv(
        "MOBILENET_V3_LARGE_PATH",
        os.path.join(ROOT_DIR, "mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth")
    )
    SMALL_MODEL_PATH = os.getenv(
        "MOBILENET_V3_SMALL_PATH",
        os.path.join(ROOT_DIR, "mobilenetv3/pretrained/mobilenetv3-small-55df8e1f.pth")
    )



# ==================== Feature Dimension Calculation ====================

def return_compressed_image_feature_dim(victim_model_name, leaked_layer_name):
    """
    Calculate the feature dimension for a given victim model and leaked layer.

    Args:
        victim_model_name (str): Name of the victim model (e.g., 'ViT-32B', 'RN50')
        leaked_layer_name (str): Name of the leaked layer (e.g., 'vit-base', 'resnet-layer1')

    Returns:
        int or list: Feature dimension(s) for the specified model and layer.
                    Returns int for 1D features, list [C, H, W] for 3D features.

    Raises:
        AssertionError: If victim model name is not supported.
        ValueError: If leaked layer name is not valid for the specified model.
    """
    # Validate victim model name
    assert victim_model_name in VICTIM_MODEL_NAMES, (
        f"Victim Model's name is invalid. Expected one of "
        f"{list(VICTIM_MODEL_NAMES.keys())}, but got {victim_model_name}"
    )

    # Vision Transformer models (ViT)
    if victim_model_name in ["ViT-32B", "ViT-16B"]:
        if leaked_layer_name == "vit-base":
            return 512  # Projected feature dimension
        elif leaked_layer_name == "vit-no-proj":
            return 768  # Raw ViT embedding dimension

    elif victim_model_name == "ViT-14L":
        if leaked_layer_name == "vit-base":
            return 1024  # Projected feature dimension for ViT-L
        elif leaked_layer_name == "vit-no-proj":
            return 1280  # Raw ViT-L embedding dimension

    # ResNet models (RN50, RN101)
    elif victim_model_name in ["RN50", "RN101"]:
        if leaked_layer_name == "resnet-base":
            # Final pooled features (different for RN50 and RN101)
            return 1024 if victim_model_name == "RN50" else 512
        elif leaked_layer_name == "resnet-layer1":
            return [256, 56, 56]  # [channels, height, width]
        elif leaked_layer_name == "resnet-layer2":
            return [512, 28, 28]
        elif leaked_layer_name == "resnet-layer3":
            return [1024, 14, 14]
        elif leaked_layer_name == "resnet-layer4":
            return [2048, 7, 7]

    # MobileNetV2
    elif victim_model_name == "mobilenetv2":
        if leaked_layer_name == "mobilenet-base":
            return 1000  # Final classification layer output
        elif leaked_layer_name == "mobilenet-all-blocks":
            return [320, 7, 7]  # Final convolutional features
        elif leaked_layer_name == "mobilenet-layer1":
            return [32, 112, 112]  # Early layer features
        elif leaked_layer_name == "mobilenet-layer-mid":
            return [64, 14, 14]  # Middle layer features
        else:
            raise ValueError(
                f"Error! There is no {leaked_layer_name} named targeted layer!!"
            )

    # MobileNetV3-Large
    elif victim_model_name == "mobilenetv3-large":
        if leaked_layer_name == "mobilenet-base":
            return 1000
        elif leaked_layer_name == "mobilenet-all-blocks":
            return [160, 7, 7]
        elif leaked_layer_name == "mobilenet-layer1":
            return [16, 112, 112]
        elif leaked_layer_name == "mobilenet-layer-mid":
            return [80, 14, 14]
        else:
            raise ValueError(
                f"Error! There is no {leaked_layer_name} named targeted layer!!"
            )

    # MobileNetV3-Small
    elif victim_model_name == "mobilenetv3-small":
        if leaked_layer_name == "mobilenet-base":
            return 1000
        elif leaked_layer_name == "mobilenet-all-blocks":
            return [96, 7, 7]
        elif leaked_layer_name == "mobilenet-layer1":
            return [16, 112, 112]
        elif leaked_layer_name == "mobilenet-layer-mid":
            return [96, 7, 7]
        else:
            raise ValueError(
                f"Error! There is no {leaked_layer_name} named targeted layer!!"
            )



# ==================== Command-line Argument Parsing ====================

def get_args():
    """
    Parse command-line arguments for training and evaluation.

    Returns:
        argparse.Namespace: Parsed arguments with the following main groups:
            - Data arguments: Dataset paths and configurations
            - Model arguments: Model architecture and device settings
            - Training arguments: Training hyperparameters
            - Evaluation arguments: Evaluation configurations
    """
    parser = argparse.ArgumentParser(
        description="Image2CaptionAttack: Feature Leakage Attack on Image Captioning"
    )

    # -------------------- Data Arguments --------------------
    data_group = parser.add_argument_group(
        title="Data Arguments",
        description="Arguments for data loading and processing"
    )
    data_group.add_argument(
        "--dataset-name",
        type=str,
        default="COCO2017",
        help="Name of the dataset (e.g., COCO2017, flickr8k, imagenet)"
    )
    data_group.add_argument(
        "--train-dir",
        type=str,
        default=os.getenv("COCO_TRAIN_DIR", "./datasets/COCO2017/train2017"),
        help="Directory containing training images"
    )
    data_group.add_argument(
        "--train-annotation-file",
        type=str,
        default=os.getenv("COCO_TRAIN_ANNOTATION", "./datasets/COCO2017/annotations/instances_train2017.json"),
        help="Path to training annotation file (COCO format)"
    )
    data_group.add_argument(
        "--val-dir",
        type=str,
        default=os.getenv("COCO_VAL_DIR", "./datasets/COCO2017/val2017"),
        help="Directory containing validation images"
    )
    data_group.add_argument(
        "--val-annotation-file",
        type=str,
        default=os.getenv("COCO_VAL_ANNOTATION", "./datasets/COCO2017/annotations/instances_val2017.json"),
        help="Path to validation annotation file (COCO format)"
    )
    data_group.add_argument(
        "--train-cap-file",
        type=str,
        default=os.getenv("COCO_TRAIN_CAPTION", "./datasets/COCO2017/annotations/captions_train2017.json"),
        help="Path to training caption file"
    )
    data_group.add_argument(
        "--val-cap-file",
        type=str,
        default=os.getenv("COCO_VAL_CAPTION", "./datasets/COCO2017/annotations/captions_val2017.json"),
        help="Path to validation caption file"
    )
    data_group.add_argument(
        "--log-dir",
        type=str,
        default=os.getenv("LOG_DIR", "./logs"),
        help="Directory to save training logs"
    )

    parser.add_argument_group(
        title="Model Arguments", description="Arguments for model configuration"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="The mode of runing QformerOPTModel"
    )
    # parser.add_argument("--model-name", type=str, default="QformerOPTModel")
    parser.add_argument("--blip-model", type=str, default="blip2-opt-2.7b")
    parser.add_argument("--victim-model", type=str, default="ViT-32B")
    parser.add_argument("--model-device", type=str, default="cuda:0")
    parser.add_argument("--victim-device", type=str, default="cuda:0")
    parser.add_argument("--leaked-feature-layer", type=str, default="vit-base")
    parser.add_argument("--add-noise", type = str, default = "False")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument_group(
        title="Training Arguments", description="Arguments for training the model"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30000,
        help="The maximum number of samples to use for training",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--intermediate-save-every", type=int, default=6)
    parser.add_argument("--log-every", type = int, default = 200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--model-save-path", type=str, default=os.getenv("MODEL_SAVE_PATH", "./saved_models/QformerOPTModel")
    )

    parser.add_argument_group(
        title="Evaluation Arguments", description="Arguments for evaluating the model"
    )
    parser.add_argument("--epochs-to-eval", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type = int, default = 50)
    parser.add_argument("--num-beams", type = int, default = 10)
    parser.add_argument("--length-penalty", type = float, default = 2.0)
    parser.add_argument("--temperature", type = float, default = 1.0)
    parser.add_argument("--do-sample", type = bool, default = True)
    parser.add_argument("--top-k", type = int, default = 10)
    parser.add_argument("--top-p", type = float, default = 0.95)
    parser.add_argument("--num-return-sequences", type = int, default = 1)
    parser.add_argument("--eval-max-samples", type=int, default=5000)
    parser.add_argument("--eval-log-every", type=int, default=200)
    parser.add_argument(
        "--eval-save-path", type=str, default=os.getenv("EVAL_SAVE_PATH", "./results")
    )

    args = parser.parse_args()

    # preprocess parameters in args
    assert args.mode in MODEL_MODES, "Model mode is invalid. Expected one of {}, but got {}".format(MODEL_MODES, args.mode)
    assert (
        args.victim_model in VICTIM_MODEL_NAMES
    ), "Victim Model did not found, expected one of {}, but got {}".format(
        list(VICTIM_MODEL_NAMES.keys()), args.victim_model
    )
    args.add_noise = True if args.add_noise == "True" else False
    assert args.add_noise == False or (args.add_noise == True and args.victim_model in ["RN50", "RN101"]), "Add noise is only supported for RN50 and RN101 models, but got {}".format(args.victim_model)
    # TODO: HERE I add clip_model to make this code compatible with its older version, after that I will totally update the code and remove the following two lines
    args.clip_model = args.victim_model
    args.clip_device = args.victim_device
    assert args.blip_model in BLIP_MODEL_NAMES, "BLIP Model did not found. Expect the model name is one of {}, but got {}".format(BLIP_MODEL_NAMES, args.blip_model)

    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
    print("Done!")
