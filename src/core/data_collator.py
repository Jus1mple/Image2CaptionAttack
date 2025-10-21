#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data collation functions for batching and preprocessing.

This module provides collate functions that combine individual dataset
samples into batches with proper padding and preprocessing.
"""

import torch
from typing import List, Dict, Any


def collate_fn(batch: List[tuple], processor) -> Dict[str, Any]:
    """
    Collate function for batching image-caption pairs.

    This function takes a list of individual samples and combines them into
    a single batch with proper padding for text and stacking for images.

    Args:
        batch: List of tuples, each containing:
            - image_id (int): Image identifier
            - raw_image (Tensor): Original image [3, H, W]
            - intermediate_features (Tensor): Leaked features from victim model
            - captions (List[str]): List of caption strings for the image
        processor: BLIP-2 processor for tokenizing captions

    Returns:
        Dictionary containing batched data:
            - image_ids (List[int]): List of image IDs
            - raw_images (Tensor): Stacked original images [B, 3, H, W]
            - intermediate_image_features (Tensor): Stacked intermediate features
            - captions (List[List[str]]): Original captions (list of lists)
            - input_ids (Tensor): Tokenized caption IDs [B, max_length]
            - attention_mask (Tensor): Attention masks [B, max_length]

    Example:
        >>> from functools import partial
        >>> batch_collate_fn = partial(collate_fn, processor=blip2_processor)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=batch_collate_fn)
    """
    processed_batch = {}

    # Extract components from batch
    image_ids = []
    raw_images = []
    intermediate_image_features = []
    captions = []  # First caption of each image for training

    for item in batch:
        image_ids.append(item[0])           # Image ID
        raw_images.append(item[1])          # Raw image tensor
        intermediate_image_features.append(item[2])  # Leaked features
        captions.append(item[3][0])         # First caption for training

    # Stack image tensors
    raw_images = torch.cat([img.unsqueeze(0) for img in raw_images], dim=0)
    intermediate_image_features = torch.cat(
        [feat.unsqueeze(0) for feat in intermediate_image_features], dim=0
    )

    # Populate processed batch
    processed_batch["image_ids"] = image_ids
    processed_batch["raw_images"] = raw_images
    processed_batch["intermediate_image_features"] = intermediate_image_features
    processed_batch["captions"] = [item[3] for item in batch]  # All captions

    # Tokenize captions with padding
    caption_inputs = processor.tokenizer(
        captions,
        padding="max_length",
        max_length=50,
        truncation=True,
        return_tensors="pt",
    )

    processed_batch["input_ids"] = caption_inputs["input_ids"]
    processed_batch["attention_mask"] = caption_inputs["attention_mask"]

    return processed_batch


def collate_fn_for_ocr(batch: List[tuple], processor) -> Dict[str, Any]:
    """
    Collate function for OCR tasks (placeholder for future implementation).

    Args:
        batch: List of tuples containing OCR data
        processor: BLIP-2 processor

    Returns:
        Dictionary containing batched OCR data

    Note:
        This is a placeholder for future OCR task support.
        Currently uses the same implementation as caption collation.
    """
    # TODO: Implement OCR-specific collation if needed
    return collate_fn(batch, processor)
