#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging utility for Image2CaptionAttack project.

This module provides a centralized logging system that outputs to both
console and file simultaneously.
"""

import logging


def setup_logger(log_file="train.log"):
    """
    Set up a logger that writes to both console and file.

    Args:
        log_file (str): Path to the log file. Defaults to "train.log".

    Returns:
        logging.Logger: Configured logger object with both file and console handlers.

    Example:
        >>> logger = setup_logger("training.log")
        >>> logger.info("Training started")
        >>> logger.warning("Learning rate decreased")
        >>> logger.error("CUDA out of memory")
    """
    # Create logger
    logger = logging.getLogger("training_logger")

    # Set logging level
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler for logging to file
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    # Test the logger
    test_logger = setup_logger("test.log")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
