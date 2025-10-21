#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module for Image2CaptionAttack project.

This module handles the evaluation/inference of the trained QFormer+OPT model,
generating captions from leaked intermediate features and saving results.
"""

import os
import torch
from tqdm import tqdm
import json


def evaluate(
    to_eval_model,
    eval_loader,
    processor,
    device,
    max_length=50,
    num_beams=10,
    length_penalty=2.0,
    temperature=1.0,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    eval_log_every=200,
    eval_save_path=None,
    add_noise=False
):
    """
    Evaluate the trained model on a test/validation dataset.

    This function generates captions from leaked intermediate features and saves
    the results in JSONL format for further evaluation.

    Args:
        to_eval_model (nn.Module): The trained QFormerOptModel to evaluate
        eval_loader (DataLoader): DataLoader for evaluation data
        processor (AutoProcessor): BLIP-2 processor for tokenization
        device (str or torch.device): Device to run evaluation on
        max_length (int): Maximum length of generated captions. Default: 50
        num_beams (int): Number of beams for beam search. Default: 10
        length_penalty (float): Length penalty for generation. Default: 2.0
        temperature (float): Sampling temperature. Default: 1.0
        do_sample (bool): Whether to use sampling. Default: True
        top_k (int): Top-k filtering parameter. Default: 50
        top_p (float): Top-p (nucleus) filtering parameter. Default: 0.95
        num_return_sequences (int): Number of sequences to return. Default: 1
        eval_log_every (int): Log progress every N batches. Default: 200
        eval_save_path (str, optional): Directory to save evaluation results
        add_noise (bool): Whether noise was added to features (for naming). Default: False

    Returns:
        list: List of dictionaries containing:
            - image_id: Image ID from the dataset
            - generated: Generated caption
            - ground_truth: Ground truth captions (list)

    Example:
        >>> model = load_model(...)
        >>> results = evaluate(
        ...     to_eval_model=model,
        ...     eval_loader=test_loader,
        ...     processor=blip2_processor,
        ...     device='cuda:0',
        ...     eval_save_path='./results'
        ... )
    """
    # Set model to evaluation mode
    to_eval_model.eval()
    to_eval_model.to(device)

    results = []

    # Prepare output file
    fout = None
    if eval_save_path is not None:
        os.makedirs(eval_save_path, exist_ok=True)
        output_filename = "eval_results_noise.jsonl" if add_noise else "eval_results.jsonl"
        fout = open(os.path.join(eval_save_path, output_filename), "w", encoding='utf-8')

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(eval_loader, total=len(eval_loader), desc="Evaluating")
        ):
            # Generate captions from intermediate features
            generated_ids = to_eval_model.generate(
                intermediate_features=batch["intermediate_image_features"].to(device),
                max_length=max_length,
                do_sample=do_sample,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
            )

            # Decode generated captions
            generated_captions = processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Get ground truth data
            ground_truth_image_ids = batch["image_ids"]
            ground_truth_captions = batch["captions"]

            # Create result pairs
            for gt_image_id, gen_caption, gt_caption in zip(
                ground_truth_image_ids, generated_captions, ground_truth_captions
            ):
                # Clean up generated caption (keep only first sentence)
                clean_caption = gen_caption.split(".")[0] + "."

                result_dict = {
                    "image_id": gt_image_id,
                    "generated": clean_caption,
                    "ground_truth": gt_caption
                }

                results.append(result_dict)

                # Save to file if path provided
                if fout is not None:
                    fout.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

            # Log progress
            if eval_log_every > 0 and (batch_idx + 1) % eval_log_every == 0:
                print(f"Processed {batch_idx + 1}/{len(eval_loader)} batches")

    # Close output file
    if fout is not None:
        fout.close()
        print(f"Evaluation results saved to: {eval_save_path}")

    return results


if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import and use the evaluate() function in your main script.")
