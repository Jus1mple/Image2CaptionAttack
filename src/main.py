#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training and evaluation script for Image2CaptionAttack.

This script trains the QFormer+OPT attack model on leaked intermediate
features from victim vision models and evaluates caption reconstruction.

Supported datasets: COCO2017, Flickr8K, ImageNet
Supported victim models: CLIP (ViT/ResNet), MobileNet variants
"""

import os
import torch
import torch.optim as optim
from functools import partial
from transformers import AutoModelForVision2Seq, AutoProcessor

# Project modules
from core import train, evaluate, collate_fn
from models import QFormerOptModel
from utils import (
    set_random_seeds,
    return_blip2_model_path,
    get_encode_fn,
    load_checkpoint,
    get_dataset_processor,
    init_intermediate_dim_and_extra_conv_module,
)
from config import VICTIM_MODEL_NAMES, get_args, _TRAIN, _TEST, _WHOLE
from logger import setup_logger

# Disable cuDNN for reproducibility
torch.backends.cudnn.enabled = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_model(args, logger):
    """
    Initialize the attack model and BLIP-2 components.

    Args:
        args: Command-line arguments
        logger: Logger instance

    Returns:
        Tuple of (model, blip2processor)
    """
    # Load BLIP-2 model and processor
    logger.info("Loading BLIP-2 model...")
    blip2_model_path = return_blip2_model_path(args.blip_model)
    blip2model = AutoModelForVision2Seq.from_pretrained(
        blip2_model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    blip2processor = AutoProcessor.from_pretrained(blip2_model_path)

    # Initialize intermediate feature dimension and conv module
    intermediate_feature_dim, extra_conv_module = (
        init_intermediate_dim_and_extra_conv_module(
            args.victim_model, args.leaked_feature_layer
        )
    )
    logger.info(f"Intermediate feature dim: {intermediate_feature_dim}")

    # Initialize QFormer+OPT attack model
    model = QFormerOptModel(
        blip2model,
        blip2vison_output_dim=1408,  # Fixed for BLIP-2
        intermediate_feature_dim=intermediate_feature_dim,
        extra_conv_module=extra_conv_module,
    )
    model.freeze_all_except_target_layers()
    model = model.to(args.model_device)

    return model, blip2processor


def setup_dataset_processor(args, victim_model_name, logger):
    """
    Create dataset processor for loading images and extracting features.

    Args:
        args: Command-line arguments
        victim_model_name: Name of victim model
        logger: Logger instance

    Returns:
        Initialized dataset processor instance
    """
    # Get dataset processor class
    DatasetProcessorClass = get_dataset_processor(args.dataset_name)
    assert DatasetProcessorClass is not None, (
        f"Invalid dataset name: {args.dataset_name}"
    )

    # Determine target image size
    target_size = 336 if "336px" in victim_model_name else 224
    logger.info(f"Target image size: {target_size}")

    # Get feature extraction function
    extract_feature_func = get_encode_fn(args.leaked_feature_layer, args.add_noise)
    assert extract_feature_func is not None, (
        f"Invalid leaked feature layer: {args.leaked_feature_layer}"
    )

    # Select data directory based on mode
    if args.mode == _TRAIN:
        data_dir = args.train_dir
        annotation_file = args.train_annotation_file
        cap_file = args.train_cap_file
    else:  # _TEST mode
        data_dir = args.val_dir
        annotation_file = args.val_annotation_file
        cap_file = args.val_cap_file

    # Initialize processor
    processor = DatasetProcessorClass(
        data_dir=data_dir,
        annotation_file=annotation_file,
        caption_file=cap_file,
        target_size=target_size,
        victim_model=VICTIM_MODEL_NAMES[victim_model_name],
        victim_device=args.victim_device,
        extract_feature_func=extract_feature_func,
        add_noise=args.add_noise,
    )

    return processor


def train_mode(args, model, blip2processor, logger):
    """
    Execute training mode.

    Args:
        args: Command-line arguments
        model: QFormerOptModel instance
        blip2processor: BLIP-2 processor for tokenization
        logger: Logger instance
    """
    logger.info("=== Training Mode ===")

    # Setup model save directory
    model_save_dir = os.path.join(
        args.model_save_path,
        f"{args.dataset_name}_{args.blip_model}_{args.victim_model}_{args.leaked_feature_layer}",
    )
    os.makedirs(model_save_dir, exist_ok=True)

    # Initialize optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # Create dataset processor
    logger.info("Creating dataset processor...")
    dataset_processor = setup_dataset_processor(args, args.victim_model, logger)

    # Create data loader
    train_dataloader = dataset_processor.get_data_loader(
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=blip2processor),
    )

    # Train model
    train(
        to_train_model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        model_save_path=model_save_dir,
        device=args.model_device,
        intermediate_save_every=args.intermediate_save_every,
        log_every=args.log_every,
        logger=logger,
        add_noise=args.add_noise,
    )


def test_mode(args, model, blip2processor, logger):
    """
    Execute evaluation/testing mode.

    Args:
        args: Command-line arguments
        model: QFormerOptModel instance
        blip2processor: BLIP-2 processor for tokenization
        logger: Logger instance
    """
    logger.info("=== Testing Mode ===")

    # Load trained model checkpoint
    epoch_folder_name = (
        f"epoch_{args.epochs_to_eval}_noise" if args.add_noise
        else f"epoch_{args.epochs_to_eval}"
    )
    checkpoint_path = os.path.join(
        args.model_save_path,
        f"{args.dataset_name}_{args.blip_model}_{args.victim_model}_{args.leaked_feature_layer}",
        epoch_folder_name,
    )

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, model, device=args.model_device)

    # Setup evaluation save directory
    eval_save_path = os.path.join(
        args.eval_save_path,
        f"{args.dataset_name}_{args.blip_model}_{args.victim_model}_{args.leaked_feature_layer}",
    )
    os.makedirs(eval_save_path, exist_ok=True)

    # Create dataset processor
    logger.info("Creating dataset processor...")
    dataset_processor = setup_dataset_processor(args, args.victim_model, logger)

    # Create data loader
    test_dataloader = dataset_processor.get_data_loader(
        batch_size=args.eval_batch_size,
        max_samples=args.eval_max_samples,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=blip2processor),
    )

    # Evaluate model
    evaluate(
        to_eval_model=model,
        eval_loader=test_dataloader,
        processor=blip2processor,
        device=args.model_device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        eval_log_every=args.eval_log_every,
        eval_save_path=eval_save_path,
        add_noise=args.add_noise,
    )


def main():
    """
    Main entry point for training and evaluation.

    Parses arguments and executes the appropriate mode:
    - train: Train attack model on leaked features
    - test: Evaluate trained model and generate captions
    - whole: End-to-end pipeline (not yet implemented)
    """
    # Parse command-line arguments
    args = get_args()

    # Set random seeds for reproducibility
    set_random_seeds(args.seed)

    # Setup logger
    log_filename = os.path.join(
        args.log_dir,
        f"{args.dataset_name}_{args.mode}_{args.blip_model}_{args.victim_model}_{args.leaked_feature_layer}.log",
    )
    logger = setup_logger(log_filename)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Image2CaptionAttack - Feature Leakage Attack")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args}")
    logger.info("=" * 60)

    # Setup model
    model, blip2processor = setup_model(args, logger)

    # Execute appropriate mode
    if args.mode == _TRAIN:
        train_mode(args, model, blip2processor, logger)

    elif args.mode == _TEST:
        test_mode(args, model, blip2processor, logger)

    elif args.mode == _WHOLE:
        # TODO: Implement end-to-end pipeline
        logger.warning("Whole mode not yet implemented!")
        raise NotImplementedError(
            "End-to-end pipeline mode is not yet implemented. "
            "Please use 'train' or 'test' mode separately."
        )

    else:
        raise ValueError(
            f"Unknown mode: {args.mode}. "
            f"Supported modes: {_TRAIN}, {_TEST}, {_WHOLE}"
        )

    logger.info("=" * 60)
    logger.info("Execution completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
