"""
Predictor class for greenhouse climate predictions.

This module provides a high-level interface for running inference
with trained greenhouse climate models.
"""

from enum import StrEnum
from types import SimpleNamespace

import torch
from pydantic import BaseModel
from torch import Tensor, inference_mode

from exoprompt_inference.inference.model_loader import ModelType, load_model_from_checkpoint
from exoprompt_inference._vendor.models.greenlight_gt_timeseries_module import (
    GreenlightGTTimeSeriesLitModule,
)


class ModelInputMode(StrEnum):
    """Input mode for the model, determining exogenous parameter handling.

    Attributes:
        VANILLA: No exogenous parameters used.
        EXO_PROMPT_254: Full 254-dimensional exogenous parameter vector.
        EXO_PROMPT_C_LEAKAGE: Subset of parameters focused on leakage coefficient.
    """

    VANILLA = "VANILLA"
    EXO_PROMPT_254 = "EXO_PROMPT_254"
    EXO_PROMPT_C_LEAKAGE = "EXO_PROMPT_C_LEAKAGE"


class _PredictionConfig(BaseModel):
    """Internal configuration derived from model for prediction."""

    pred_len: int
    label_len: int
    use_all_features_for_decoder: bool
    output_feature_idx: list[int] | None
    feature_idx_for_decoder: list[int] | None
    uses_exo_prompt: bool = False


class Predictor:
    """
    High-level predictor for greenhouse climate models.

    Wraps a trained PyTorch Lightning model and provides a simple interface
    for making predictions. Handles decoder input construction, device
    management, and output extraction.

    Args:
        model: Trained Lightning module (GreenlightGTTimeSeriesLitModule).
        model_input_mode: How exogenous parameters should be handled.

    Example:
        >>> from exoprompt_inference.inference import load_model_from_checkpoint, ModelType
        >>> model = load_model_from_checkpoint("model.ckpt", ModelType.TIME_SERIES_LIB_MODEL)
        >>> predictor = Predictor(model, ModelInputMode.EXO_PROMPT_254)
        >>> predictions = predictor.predict(seq_x, seq_x_mark, seq_y_mark, exo_prompt)

    Note:
        The model's device and dtype are automatically detected and used for all
        internal tensor operations.
    """

    def __init__(
            self,
            model: GreenlightGTTimeSeriesLitModule,
            model_input_mode: ModelInputMode,
    ):
        self.model = model
        self.model_input_mode = model_input_mode
        self._pred_config = self._extract_prediction_config()

    @property
    def device(self) -> torch.device:
        """Device where the model is located."""
        return self.model.device

    @property
    def pred_len(self) -> int:
        """Prediction horizon length."""
        return self._pred_config.pred_len

    @property
    def label_len(self) -> int:
        """Label length (decoder lookback)."""
        return self._pred_config.label_len

    @property
    def uses_exo_prompt(self) -> bool:
        return self._pred_config.uses_exo_prompt

    def _extract_prediction_config(self) -> _PredictionConfig:
        """Extract prediction configuration from model config."""
        model_configs: SimpleNamespace = self.model.model_configs

        pred_len = model_configs.pred_len
        label_len = getattr(model_configs, "label_len", 0)

        use_all_features_for_decoder = getattr(
            model_configs, "use_all_features_for_decoder", False
        )

        # Default output indices are [0, 1, 2] for tAir, vpAir, co2Air
        # after being extracted from full feature set at [7, 8, 9]
        output_feature_idx = getattr(model_configs, "output_feature_idx", None)
        if output_feature_idx is None:
            output_feature_idx = [0, 1, 2]

        feature_idx_for_decoder = None
        if not use_all_features_for_decoder:
            # Indoor climate features (tAir, vpAir, co2Air) at indices 7, 8, 9
            feature_idx_for_decoder = [7, 8, 9]

        enable_exo_prompt_tuning = getattr(model_configs, "enable_exo_prompt_tuning", False)

        return _PredictionConfig(
            pred_len=pred_len,
            label_len=label_len,
            use_all_features_for_decoder=use_all_features_for_decoder,
            output_feature_idx=output_feature_idx,
            feature_idx_for_decoder=feature_idx_for_decoder,
            uses_exo_prompt=enable_exo_prompt_tuning == True,
        )

    def _build_decoder_input(self, seq_x: Tensor) -> Tensor:
        """
        Build decoder input from encoder sequence.

        For transformer-style models, the decoder input consists of:
        - The last `label_len` timesteps from seq_x (as context)
        - Zero-padded `pred_len` timesteps (to be predicted)

        Args:
            seq_x: Encoder input sequence [batch, seq_len, features]

        Returns:
            Decoder input tensor [batch, label_len + pred_len, features]
        """
        batch_size = seq_x.shape[0]
        device = seq_x.device
        dtype = seq_x.dtype

        if self._pred_config.use_all_features_for_decoder:
            num_features = seq_x.shape[-1]
            # Get label portion from seq_x
            if self._pred_config.label_len > 0:
                label_portion = seq_x[:, -self._pred_config.label_len:, :]
            else:
                label_portion = torch.empty(batch_size, 0, num_features, device=device, dtype=dtype)

            # Zero-pad for prediction portion
            pred_portion = torch.zeros(
                batch_size, self._pred_config.pred_len, num_features, device=device, dtype=dtype
            )
        else:
            # Use only specific features for decoder (indoor climate)
            feature_idx = self._pred_config.feature_idx_for_decoder
            num_features = len(feature_idx)

            if self._pred_config.label_len > 0:
                label_portion = seq_x[:, -self._pred_config.label_len:, feature_idx]
            else:
                label_portion = torch.empty(batch_size, 0, num_features, device=device, dtype=dtype)

            pred_portion = torch.zeros(
                batch_size, self._pred_config.pred_len, num_features, device=device, dtype=dtype
            )

        return torch.cat([label_portion, pred_portion], dim=1)

    @inference_mode()
    def predict(
            self,
            seq_x: Tensor,
            seq_x_mark: Tensor,
            seq_y_mark: Tensor,
            exo_prompt: Tensor | None = None,
    ) -> Tensor:
        """
        Make predictions for greenhouse climate.

        Args:
            seq_x: Input sequence [batch, seq_len, num_features]
                   Scaled time series data with 18 features.
            seq_x_mark: Encoder time features [batch, seq_len, num_time_features]
                        Typically 5 time features (month, day, weekday, hour, minute).
            seq_y_mark: Decoder time features [batch, label_len + pred_len, num_time_features]
                        Time features for decoder context and prediction horizon.
            exo_prompt: Optional exogenous parameters [batch, exo_params_len]
                        Scaled greenhouse model parameters (e.g., 254 dimensions).

        Returns:
            Predictions tensor [batch, pred_len, num_outputs]
            Where num_outputs is typically 3 (tAir, vpAir, co2Air).

        Note:
            - All input tensors should be on the same device as the model.
            - Inputs should be scaled using the same scaler as training data.
            - Use `GreenlightInferenceDataset.inverse_transform()` to convert
              predictions back to original scale.
        """
        # Ensure inputs are on the correct device
        device = self.device
        seq_x = seq_x.to(device)
        seq_x_mark = seq_x_mark.to(device)
        seq_y_mark = seq_y_mark.to(device)
        if exo_prompt is not None:
            exo_prompt = exo_prompt.to(device)

        # Build decoder input
        dec_inp = self._build_decoder_input(seq_x)

        # Forward pass
        outputs = self.model.forward(
            x=(seq_x, seq_x_mark, dec_inp, seq_y_mark),
            exo_prompt=exo_prompt,
        )

        # Extract only the prediction portion
        outputs = outputs[:, -self._pred_config.pred_len:, :]

        # Extract output features if specified
        if self._pred_config.output_feature_idx is not None:
            outputs = outputs[:, :, self._pred_config.output_feature_idx]

        return outputs
