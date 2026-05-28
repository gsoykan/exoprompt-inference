"""
Inference utilities for greenhouse climate prediction.

This module provides model loading and prediction utilities for
running inference with trained greenhouse climate models.

Example:
    >>> from exoprompt_inference.data.inference import (
    ...     load_model_from_checkpoint,
    ...     ModelType,
    ...     Predictor,
    ...     ModelInputMode,
    ... )
    >>> model = load_model_from_checkpoint(
    ...     "model.ckpt",
    ...     ModelType.TIME_SERIES_LIB_MODEL,
    ...     device="cuda",
    ... )
    >>> predictor = Predictor(model, ModelInputMode.VANILLA)
    >>> predictions = predictor.predict(seq_x, seq_x_mark, seq_y_mark)
"""

from exoprompt_inference.inference.model_loader import (
    ModelType,
    load_model_from_checkpoint,
)
from exoprompt_inference.inference.predictor import (
    ModelInputMode,
    Predictor,
)

__all__ = [
    # Model loading
    "load_model_from_checkpoint",
    "ModelType",
    # Prediction
    "Predictor",
    "ModelInputMode",
]
