"""
Model loading utilities for greenhouse climate prediction.

This module provides functions to load trained PyTorch Lightning models
from checkpoints for inference.
"""

import contextlib
from enum import StrEnum
from pathlib import Path

import torch

from exoprompt_inference._vendor.models.greenlight_gt_timeseries_module import (
    GreenlightGTTimeSeriesLitModule,
)


class ModelType(StrEnum):
    """Type of model architecture to load.

    Attributes:
        TIME_SERIES_LIB_MODEL: Transformer backbone (vanilla or with ExoPrompt tuning).
    """

    TIME_SERIES_LIB_MODEL = "TIME_SERIES_LIB_MODEL"


def load_model_from_checkpoint(
        checkpoint: str | Path,
        model_type: ModelType = ModelType.TIME_SERIES_LIB_MODEL,
        device: torch.device | str = "cpu",
) -> GreenlightGTTimeSeriesLitModule:
    """
    Load a trained model from checkpoint for inference.

    Args:
        checkpoint: Path to the .ckpt checkpoint file.
        model_type: Type of model architecture (only TIME_SERIES_LIB_MODEL is supported
            in this demo).
        device: Device to load the model onto ('cpu', 'cuda', 'mps', or torch.device).

    Returns:
        Loaded model in eval mode, ready for inference.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If model_type is unknown.

    Example:
        >>> model = load_model_from_checkpoint(
        ...     "path/to/checkpoint.ckpt",
        ...     ModelType.TIME_SERIES_LIB_MODEL,
        ...     device="cpu",
        ... )
        >>> output = model(input_data)
    """
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    match model_type:
        case ModelType.TIME_SERIES_LIB_MODEL:
            lit_module = GreenlightGTTimeSeriesLitModule.load_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                pretrained_ckpt=None,
            )
        case _:
            raise ValueError(f"Unknown model_type: {model_type}")

    lit_module.to(device)
    lit_module.eval()
    return lit_module
