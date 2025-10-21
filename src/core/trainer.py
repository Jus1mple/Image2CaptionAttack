#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for Image2CaptionAttack project.

This module handles the training loop for the QFormer+OPT model that learns
to reconstruct image captions from leaked intermediate features.
"""

import os
import torch
from tqdm import tqdm
from utils import save_modules


def train(
    to_train_model,
    train_dataloader,
    optimizer,
    num_epochs,
    model_save_path,
    device,
    intermediate_save_every=2,
    log_every=200,
    logger=None,
    add_noise=False
):
    """
    Train the QFormer+OPT model on leaked features to generate captions.

    Args:
        to_train_model (nn.Module): The QFormerOptModel to train
        train_dataloader (DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW)
        num_epochs (int): Number of training epochs
        model_save_path (str): Directory to save model checkpoints
        device (str or torch.device): Device to train on (e.g., 'cuda:0')
        intermediate_save_every (int): Save checkpoint every N epochs. Default: 2
        log_every (int): Log training loss every N steps. Default: 200
        logger (logging.Logger, optional): Logger for training progress
        add_noise (bool): Whether noise was added to features (for naming). Default: False

    Returns:
        None

    Example:
        >>> model = QFormerOptModel(...)
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        >>> train(model, train_loader, optimizer, num_epochs=6, ...)
    """
    # Set model to training mode
    to_train_model.train()
    to_train_model.to(device)

    global_step = 0

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epoch", total=num_epochs):
        epoch_loss = 0.0
        to_train_model.train()

        # Iterate over batches
        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc="Batch", total=len(train_dataloader))
        ):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = to_train_model(
                intermediate_features=batch["intermediate_image_features"].to(device, torch.float),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
                return_dict=True,
            )

            # Compute loss and backpropagate
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            global_step += 1

            # Log training progress
            if global_step % log_every == 0 and logger is not None:
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f}")

        # Compute average epoch loss
        epoch_loss /= len(train_dataloader)

        if logger is not None:
            logger.info(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {epoch_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % intermediate_save_every == 0 or epoch == num_epochs - 1:
            # Create checkpoint folder name
            epoch_folder_name = f"epoch_{epoch + 1}"
            if add_noise:
                epoch_folder_name += "_noise"

            checkpoint_path = os.path.join(model_save_path, epoch_folder_name)
            save_modules(to_train_model, checkpoint_path)

            if logger is not None:
                logger.info(f"Model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    # Example usage
    print("This module is not meant to be run directly.")
    print("Import and use the train() function in your main script.")
