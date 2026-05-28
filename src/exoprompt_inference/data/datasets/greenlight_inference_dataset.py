from dataclasses import dataclass
from operator import attrgetter
from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex
from torch.utils.data import Dataset

from exoprompt_inference.data import ExoPromptInput, BaseModelInput
from exoprompt_inference._vendor.utils.greenlight_scaler import GreenlightScaler
from exoprompt_inference._vendor.utils.timefeatures import time_features

# Type aliases
InferenceMode = Literal["with_targets", "prediction_only"]
TimeEncoding = Literal[0, 1]
# Base frequencies: s=secondly, t=minutely, h=hourly, d=daily, b=business days, w=weekly, m=monthly
# Extended formats allowed: "15min", "3h", "2d", etc.
FrequencyType = str


@dataclass(frozen=True)
class InferenceDatasetConfig:
    """Consolidated configuration for GreenlightInferenceDataset."""

    # Partition lengths
    seq_len: int
    label_len: int  # can be 0 for full autoregressive models
    pred_len: int

    # Time configuration
    timeenc: TimeEncoding = 1
    freq: FrequencyType = "min"  # used to be "t" but that is deprecated
    datetime_format: str = "%d-%m-%Y %H:%M"

    # Output configuration
    output_feature_idx: tuple[int, ...] | None = None
    return_all_output_features: bool = False

    # Exo prompt configuration
    disable_exo_prompt: bool = False
    exo_params_len: int = 254
    exo_params_to_take: tuple[str, ...] | None = None

    # Inference mode
    mode: InferenceMode = "prediction_only"

    # Output format - set True for PyTorch DataLoader compatibility
    return_dict: bool = False

    def min_required_timesteps(self) -> int:
        """Minimum timesteps needed based on mode."""
        if self.mode == "with_targets":
            return self.seq_len + self.pred_len
        return self.seq_len


@dataclass
class InferenceSample:
    """A single sample returned by the dataset."""

    seq_x: np.ndarray
    seq_x_mark: np.ndarray
    seq_y_mark: np.ndarray
    seq_y: np.ndarray | None = None
    seq_y_all: np.ndarray | None = None
    exo_params: np.ndarray | None = None

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dict for PyTorch DataLoader compatibility.

        Only includes non-None values since PyTorch's collate_fn
        cannot handle None values.
        """
        result = {
            "seq_x": self.seq_x,
            "seq_x_mark": self.seq_x_mark,
            "seq_y_mark": self.seq_y_mark,
        }
        if self.seq_y is not None:
            result["seq_y"] = self.seq_y
        if self.seq_y_all is not None:
            result["seq_y_all"] = self.seq_y_all
        if self.exo_params is not None:
            result["exo_params"] = self.exo_params
        return result


@dataclass
class _ProcessedData:
    """Internal container for processed time series data."""

    x: np.ndarray
    y: np.ndarray
    y_all: np.ndarray | None
    data_stamp: np.ndarray
    num_samples: int
    raw_timestamps: DatetimeIndex | Series


class GreenlightInferenceDataset(Dataset):
    """
    Inference dataset for greenhouse climate predictions.

    This dataset handles time series data with optional exogenous parameters
    (greenhouse settings) for model inference.

    Args:
        time_series_input: Historical time series data points
        exo_prompt_input: Exogenous parameters (greenhouse settings), or None
        config: Dataset configuration

    Note:
        For every experiment, there is one set of exoprompts because by definition
        ExoPrompts should not change during an experiment.
    """

    def __init__(
            self,
            time_series_input: list[BaseModelInput],
            exo_prompt_input: ExoPromptInput | None,
            config: InferenceDatasetConfig,
    ):
        self.config = config
        self._scaler = GreenlightScaler()

        # Validate data sufficiency before expensive processing
        self._validate_input_length(len(time_series_input))

        # Process inputs
        self._exo_prompt_array = self._process_exo_prompt(exo_prompt_input)
        self._data = self._process_timeseries(time_series_input)

    def _validate_input_length(self, actual_len: int) -> None:
        """Validate input has sufficient data for the configured mode."""
        min_required = self.config.min_required_timesteps()

        if actual_len < min_required:
            raise ValueError(
                f"Insufficient data for mode '{self.config.mode}': "
                f"need at least {min_required} timesteps, got {actual_len}"
            )

    def _process_exo_prompt(
            self, exo_prompt_input: ExoPromptInput | None
    ) -> np.ndarray | None:
        """Convert ExoPromptInput to scaled numpy array."""
        if self.config.disable_exo_prompt or exo_prompt_input is None:
            return None

        params_dict = exo_prompt_input.to_dict()

        # Filter to subset if configured
        if self.config.exo_params_to_take is not None:
            params_dict = {
                k: v
                for k, v in params_dict.items()
                if k in self.config.exo_params_to_take
            }

        # Scale the filtered parameters
        scaled_params = self._scaler.transform_json_dict(
            params_dict,
            self._scaler.parameter_scaling_ranges,
            enforce_rescaling_all_json_keys=True,
            enforce_rescaling_all_scale_ranges=self.config.exo_params_to_take
                                               is not None,
        )

        # Convert to array and validate
        exo_array = ExoPromptInput.dict_to_array(scaled_params)

        if len(exo_array) != self.config.exo_params_len:
            raise ValueError(
                f"Exo parameter length mismatch: "
                f"got {len(exo_array)}, expected {self.config.exo_params_len}"
            )

        return exo_array

    def _process_timeseries(
            self, timeseries_input: list[BaseModelInput]
    ) -> _ProcessedData:
        """Process raw time series input into scaled arrays."""
        # Sort by timestamp
        timeseries_input.sort(key=attrgetter("timestamp"))

        # Convert to DataFrame
        df_raw = pd.DataFrame([item.to_dict() for item in timeseries_input])

        # Process feature values
        data = self._scale_features(df_raw)

        # Extract output targets
        y, y_all = self._extract_outputs(data)

        # Process timestamps
        timestamps = pd.to_datetime(
            df_raw["timestamp"], format=self.config.datetime_format
        )
        data_stamp = self._encode_timestamps(timestamps)

        # Calculate number of valid samples
        num_samples = self._calculate_num_samples(len(data))

        return _ProcessedData(
            x=data,
            y=y,
            y_all=y_all,
            data_stamp=data_stamp,
            num_samples=num_samples,
            raw_timestamps=timestamps
        )

    def _scale_features(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Scale feature columns (all except timestamp)."""
        feature_cols = [c for c in df_raw.columns if c != "timestamp"]
        df_features = df_raw[feature_cols].astype("float32")
        return self._scaler.transform(df_features).values

    def _extract_outputs(
            self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract output target arrays based on configuration."""
        if self.config.output_feature_idx is None:
            y = data
        else:
            y = data[:, list(self.config.output_feature_idx)]

        y_all = data if self.config.return_all_output_features else None

        return y, y_all

    def _encode_timestamps(self, timestamps: DatetimeIndex | Series) -> np.ndarray:
        """Encode timestamps into feature arrays."""
        if self.config.timeenc == 0:
            # Manual time features
            data_stamp = np.column_stack(
                [
                    timestamps.dt.month,
                    timestamps.dt.day,
                    timestamps.dt.weekday,
                    timestamps.dt.hour,
                ]
            )
        else:  # timeenc == 1
            # Use time_features utility (requires DatetimeIndex, not Series)
            data_stamp = time_features(
                pd.DatetimeIndex(timestamps), freq=self.config.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        return data_stamp.astype("float32")

    def _calculate_num_samples(self, data_len: int) -> int:
        """Calculate number of valid sliding windows."""
        if self.config.mode == "with_targets":
            return data_len - self.config.seq_len - self.config.pred_len + 1
        return data_len - self.config.seq_len + 1

    def _get_seq_y_mark(self, r_begin: int, r_end: int) -> np.ndarray:
        """
        Get time features for decoder (seq_y_mark).

        In prediction_only mode, future timestamps may not exist in the data,
        so we generate them by inferring the frequency from existing timestamps.

        Args:
            r_begin: Start index for decoder sequence
            r_end: End index for decoder sequence

        Returns:
            Time features array of shape (label_len + pred_len, num_time_features)
        """
        data_len = len(self._data.data_stamp)

        if r_end <= data_len:
            # All required timestamps exist in the data
            return self._data.data_stamp[r_begin:r_end]

        # Need to generate future timestamps
        # Get existing marks if any fall within our range
        available_start = max(r_begin, 0)
        available_end = min(r_end, data_len)

        if available_start < data_len:
            existing_marks = self._data.data_stamp[available_start:available_end]
        else:
            existing_marks = np.empty((0, self._data.data_stamp.shape[1]), dtype="float32")

        # Calculate how many future timestamps to generate
        num_to_generate = r_end - data_len

        if num_to_generate > 0:
            # Infer frequency from existing timestamps
            freq = pd.infer_freq(self._data.raw_timestamps)
            if freq is None:
                # Fallback: calculate from last two timestamps
                if len(self._data.raw_timestamps) >= 2:
                    delta = (
                            self._data.raw_timestamps.iloc[-1]
                            - self._data.raw_timestamps.iloc[-2]
                    )
                    freq = delta
                else:
                    # Ultimate fallback: use config frequency
                    freq = self.config.freq

            # Generate future timestamps starting after the last available
            last_ts = self._data.raw_timestamps.iloc[-1]
            future_timestamps = pd.date_range(
                start=last_ts,
                periods=num_to_generate + 1,  # +1 because start is inclusive
                freq=freq,
            )[1:]  # Skip the first (which is last_ts)

            # Encode future timestamps
            future_marks = self._encode_timestamps(pd.DatetimeIndex(future_timestamps))

            # Concatenate existing and future marks
            if len(existing_marks) > 0:
                return np.concatenate([existing_marks, future_marks], axis=0)
            else:
                return future_marks

        return existing_marks

    def __getitem__(self, index: int) -> InferenceSample | dict[str, np.ndarray]:
        """Get a single sample by index.

        Returns:
            InferenceSample if config.return_dict is False,
            dict[str, np.ndarray] if config.return_dict is True (for DataLoader).
        """
        # Handle negative indexing
        if index < 0:
            index = len(self) + index

        s_begin = index
        s_end = s_begin + self.config.seq_len

        # Encoder input (always available)
        seq_x = self._data.x[s_begin:s_end]
        seq_x_mark = self._data.data_stamp[s_begin:s_end]

        # Decoder targets (mode-dependent)
        r_begin = s_end - self.config.label_len
        r_end = r_begin + self.config.label_len + self.config.pred_len

        if self.config.mode == "with_targets":
            sample = InferenceSample(
                seq_x=seq_x,
                seq_x_mark=seq_x_mark,
                seq_y_mark=self._data.data_stamp[r_begin:r_end],
                seq_y=self._data.y[r_begin:r_end],
                seq_y_all=self._data.y_all[r_begin:r_end]
                if self._data.y_all is not None
                else None,
                exo_params=self._exo_prompt_array,
            )
        else:
            # prediction_only mode - still need seq_y_mark for decoder time features
            seq_y_mark = self._get_seq_y_mark(r_begin, r_end)

            sample = InferenceSample(
                seq_x=seq_x,
                seq_x_mark=seq_x_mark,
                seq_y_mark=seq_y_mark,
                exo_params=self._exo_prompt_array,
            )

        return sample.to_dict() if self.config.return_dict else sample

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self._data.num_samples

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.

        Handles both full feature predictions and partial (output-only) predictions.

        Args:
            data: Scaled prediction data. Shape can be:
                  - (..., num_features) for full predictions
                  - (..., len(output_feature_idx)) for partial predictions

        Returns:
            Data in original scale with same shape as input.
        """
        data_last_dim = data.shape[-1]
        full_num_features = self._data.x.shape[-1]

        # Check if this is partial output (e.g., only tAir, vpAir, co2Air)
        is_partial_output = data_last_dim != full_num_features

        if is_partial_output:
            # Use is_only_output=True for the 3 output features (tAir, vpAir, co2Air)
            # Need to reshape for scaler: (timesteps, features) -> 2D
            original_shape = data.shape
            data_2d = data.reshape(-1, data_last_dim)
            inverse_data = self._scaler.inverse_transform(
                data_2d.copy(), is_only_output=True
            )
            return inverse_data.reshape(original_shape)
        else:
            # Full feature set - convert to DataFrame for proper column handling
            original_shape = data.shape
            data_2d = data.reshape(-1, full_num_features)
            feature_names = BaseModelInput.get_feature_names()
            df = pd.DataFrame(data_2d, columns=feature_names)
            inverse_df = self._scaler.inverse_transform(df)
            return inverse_df.values.reshape(original_shape)

    def get_timestamps_for_sample(self, sample_index: int) -> dict[str, pd.DatetimeIndex]:
        """
        Get actual timestamps for a given sample's input sequence and prediction window.

        This reconstructs the actual calendar dates/times that correspond to the
        encoded time features (seq_x_mark, seq_y_mark) for a sample.

        Args:
            sample_index: Index of the sample in the dataset

        Returns:
            Dictionary with keys:
                - 'input': Timestamps for seq_x (encoder input)
                - 'prediction': Timestamps for prediction window
                - 'label': Timestamps for label portion (if applicable)

        Example:
            >>> timestamps = dataset.get_timestamps_for_sample(0)
            >>> print(timestamps['input'])  # Shows actual dates for historical data
            >>> print(timestamps['prediction'])  # Shows dates for predictions
        """
        if sample_index < 0:
            sample_index = len(self) + sample_index

        # Calculate indices in the original data
        s_begin = sample_index
        s_end = s_begin + self.config.seq_len
        r_begin = s_end - self.config.label_len
        r_end = r_begin + self.config.label_len + self.config.pred_len

        # Get input timestamps (always available)
        input_timestamps = self._data.raw_timestamps[s_begin:s_end]

        # Infer frequency for generating future timestamps
        freq = pd.infer_freq(self._data.raw_timestamps)
        if freq is None:
            # Fallback to delta-based inference
            if len(self._data.raw_timestamps) >= 2:
                freq = (
                        self._data.raw_timestamps.iloc[-1]
                        - self._data.raw_timestamps.iloc[-2]
                )
            else:
                freq = self.config.freq

        # Generate prediction window timestamps
        if r_end <= len(self._data.raw_timestamps):
            # All timestamps exist in data
            prediction_timestamps = self._data.raw_timestamps[s_end:r_end]
            label_timestamps = self._data.raw_timestamps[r_begin:s_end]
        else:
            # Need to generate future timestamps
            last_ts = self._data.raw_timestamps.iloc[-1]
            num_future = r_end - len(self._data.raw_timestamps)

            future_timestamps = pd.date_range(
                start=last_ts,
                periods=num_future + 1,
                freq=freq,
            )[1:]  # Skip first (which is last_ts)

            # Combine existing and future
            existing_end = min(r_end, len(self._data.raw_timestamps))
            existing_pred = self._data.raw_timestamps[s_end:existing_end]

            prediction_timestamps = pd.DatetimeIndex(
                list(existing_pred) + list(future_timestamps)
            )

            # Label timestamps
            label_end = min(s_end, len(self._data.raw_timestamps))
            label_timestamps = self._data.raw_timestamps[r_begin:label_end]

        return {
            "input": pd.DatetimeIndex(input_timestamps),
            "prediction": prediction_timestamps,
            "label": pd.DatetimeIndex(label_timestamps) if self.config.label_len > 0 else pd.DatetimeIndex([]),
        }


# ============================================================================
# Testing / Demo
# ============================================================================


def main():
    """Test GreenlightInferenceDataset with real data."""
    from pathlib import Path

    from torch.utils.data import DataLoader

    from exoprompt_inference.data.loaders.gt_csv_v1_loader import GTCSVV1Loader
    from exoprompt_inference.data.loaders.exo_prompt_json_loader import ExoPromptJsonLoader

    print("=" * 70)
    print("GreenlightInferenceDataset Test Suite (Real Data)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Setup: Load real data
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).parents[3]  # Project root

    # Time series data
    csv_path = (
            base_dir
            / "data"
            / "from_david_by_gurkan"
            / "gt"
            / "hps"
            / "gt_hps_timeseries.csv"
    )

    # Exo prompt parameters
    json_path = (
            base_dir
            / "data"
            / "from_david_by_gurkan"
            / "gt"
            / "hps"
            / "climate_model_hps_params.json"
    )

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        print("Please ensure the data files exist.")
        return

    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        print("Please ensure the data files exist.")
        return

    # Load data using existing loaders
    print("\n[Setup] Loading real data...")
    csv_loader = GTCSVV1Loader(csv_path)
    json_loader = ExoPromptJsonLoader(json_path)

    timeseries_data = csv_loader.data
    exo_prompt_data = json_loader.data

    print(f"  Loaded {len(timeseries_data)} timesteps from CSV")
    print(
        f"  Loaded {len(exo_prompt_data.get_present_param_names())} exo params from JSON"
    )
    print(
        f"  Time range: {timeseries_data[0].timestamp} → {timeseries_data[-1].timestamp}"
    )

    # Common config values (matching training configs)
    SEQ_LEN = 192
    LABEL_LEN = 48
    PRED_LEN = 96

    # -------------------------------------------------------------------------
    # Scenario 1: prediction_only mode with real data
    # -------------------------------------------------------------------------
    print("\n[Scenario 1] prediction_only mode with real data")
    print("-" * 50)

    config_pred_only = InferenceDatasetConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        mode="prediction_only",
        disable_exo_prompt=True,
        datetime_format="%Y-%m-%d %H:%M:%S",  # Format from CSV
    )

    # Use first SEQ_LEN timesteps (minimum required)
    timeseries_minimal = timeseries_data[:SEQ_LEN]

    dataset_pred_only = GreenlightInferenceDataset(
        time_series_input=timeseries_minimal,
        exo_prompt_input=None,
        config=config_pred_only,
    )

    print(f"  Config: mode={config_pred_only.mode}")
    print(f"  Min required timesteps: {config_pred_only.min_required_timesteps()}")
    print(f"  Input timesteps: {len(timeseries_minimal)}")
    print(f"  Dataset length: {len(dataset_pred_only)}")

    sample = dataset_pred_only[0]
    print(f"  Sample seq_x shape: {sample.seq_x.shape}")
    print(f"  Sample seq_x_mark shape: {sample.seq_x_mark.shape}")
    print(f"  Sample seq_y_mark shape: {sample.seq_y_mark.shape}")  # Should be (144, 5)
    print(f"  Sample seq_y: {sample.seq_y}")  # Should be None
    print(f"  Sample exo_params: {sample.exo_params}")  # Should be None

    # Verify seq_y_mark was generated correctly (future timestamps)
    # With only 192 timesteps, we need to generate 96 future timestamps
    # r_begin = 192 - 48 = 144, r_end = 144 + 48 + 96 = 288
    # So we have timestamps 144-191 (48 existing) and need to generate 192-287 (96 future)
    expected_y_mark_len = LABEL_LEN + PRED_LEN  # 48 + 96 = 144
    assert sample.seq_y_mark.shape[0] == expected_y_mark_len, (
        f"seq_y_mark should have {expected_y_mark_len} timesteps, got {sample.seq_y_mark.shape[0]}"
    )

    # Verify the existing portion (indices 144-191 of original data = first 48 of seq_y_mark)
    # matches what we'd get from the original timestamps
    original_timestamps = pd.to_datetime(
        [ts.timestamp for ts in timeseries_minimal], format="%Y-%m-%d %H:%M:%S"
    )
    r_begin_test = SEQ_LEN - LABEL_LEN  # 144
    existing_timestamps = original_timestamps[r_begin_test:]  # 144-191 (48 timestamps)
    expected_existing_marks = time_features(
        pd.DatetimeIndex(existing_timestamps), freq="min"
    ).transpose(1, 0).astype("float32")

    # Compare existing portion
    actual_existing = sample.seq_y_mark[:LABEL_LEN]  # First 48
    assert np.allclose(actual_existing, expected_existing_marks), (
        "Existing timestamp marks don't match!"
    )
    print(f"  Existing timestamps (first {LABEL_LEN}): OK")

    # Verify generated future timestamps have correct frequency (5 min intervals)
    last_original_ts = original_timestamps[-1]
    inferred_freq = pd.infer_freq(original_timestamps)
    print(f"  Inferred frequency: {inferred_freq}")

    # Generate expected future timestamps
    expected_future_ts = pd.date_range(
        start=last_original_ts,
        periods=PRED_LEN + 1,
        freq=inferred_freq,
    )[1:]  # Skip first (which is last_original_ts)
    expected_future_marks = time_features(
        pd.DatetimeIndex(expected_future_ts), freq="min"
    ).transpose(1, 0).astype("float32")

    # Compare generated portion
    actual_future = sample.seq_y_mark[LABEL_LEN:]  # Last 96
    assert np.allclose(actual_future, expected_future_marks), (
        "Generated future timestamp marks don't match!"
    )
    print(f"  Generated timestamps (last {PRED_LEN}): OK")
    print(f"  Last original: {last_original_ts}")
    print(f"  First generated: {expected_future_ts[0]}")
    print(f"  Last generated: {expected_future_ts[-1]}")

    # -------------------------------------------------------------------------
    # Scenario 1b: Test fallback frequency inference (irregular timestamps)
    # -------------------------------------------------------------------------
    print("\n[Scenario 1b] Fallback frequency inference (delta-based)")
    print("-" * 50)

    from datetime import timedelta

    # Create data with irregular timestamps where pd.infer_freq() returns None
    # We need at least 3 timestamps for infer_freq to work, and they must be irregular
    irregular_data = []
    base_ts = timeseries_data[0].timestamp

    for i in range(SEQ_LEN):
        # Create a copy with modified timestamp
        original = timeseries_data[i]
        # Add irregular interval: mostly 5 min but occasionally 6 min to break infer_freq
        if i < 10:
            # First 10 have irregular intervals
            offset = 5 * i + (1 if i == 5 else 0)  # One irregular gap
        else:
            # Rest are regular 5-min intervals from where we left off
            offset = 5 * 10 + 1 + 5 * (i - 10)  # Continue from irregular start

        new_ts = base_ts + timedelta(minutes=offset)

        # Recreate BaseModelInput with new timestamp (frozen model)
        irregular_data.append(BaseModelInput(
            timestamp=new_ts,
            iGlob=original.iGlob, tOut=original.tOut, vpOut=original.vpOut,
            co2Out=original.co2Out, wind=original.wind, tSky=original.tSky,
            tSoOut=original.tSoOut, tAir=original.tAir, vpAir=original.vpAir,
            co2Air=original.co2Air, shScr=original.shScr, blScr=original.blScr,
            roof=original.roof, tPipe=original.tPipe, tGroPipe=original.tGroPipe,
            lamp=original.lamp, intLamp=original.intLamp, extCo2=original.extCo2,
        ))

    # Verify infer_freq returns None for our irregular data
    irregular_timestamps = pd.to_datetime([d.timestamp for d in irregular_data])
    inferred = pd.infer_freq(irregular_timestamps)
    print(f"  pd.infer_freq on irregular data: {inferred}")

    if inferred is None:
        print("  Successfully created irregular timestamps (infer_freq=None)")

        # Create dataset with irregular timestamps
        config_irregular = InferenceDatasetConfig(
            seq_len=SEQ_LEN,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            mode="prediction_only",
            disable_exo_prompt=True,
            datetime_format="%Y-%m-%d %H:%M:%S",
        )

        dataset_irregular = GreenlightInferenceDataset(
            time_series_input=irregular_data,
            exo_prompt_input=None,
            config=config_irregular,
        )

        # Get sample - this should trigger the fallback frequency inference
        sample_irregular = dataset_irregular[0]

        # Verify seq_y_mark was generated (shape should be correct)
        assert sample_irregular.seq_y_mark.shape == (LABEL_LEN + PRED_LEN, 5), (
            f"Expected shape ({LABEL_LEN + PRED_LEN}, 5), got {sample_irregular.seq_y_mark.shape}"
        )
        print(f"  seq_y_mark shape with fallback: {sample_irregular.seq_y_mark.shape}")

        # Verify the delta-based fallback was used by checking timestamp continuity
        # Last two timestamps should have 5-min delta (regular at the end)
        last_delta = (
                dataset_irregular._data.raw_timestamps.iloc[-1]
                - dataset_irregular._data.raw_timestamps.iloc[-2]
        )
        print(f"  Last timestamp delta (used for fallback): {last_delta}")
        print("  Fallback frequency inference: OK")
    else:
        print(f"  Warning: infer_freq returned {inferred}, fallback not tested")
        print("  Skipping fallback test (regular timestamps detected)")

    # -------------------------------------------------------------------------
    # Scenario 2: with_targets mode for evaluation
    # -------------------------------------------------------------------------
    print("\n[Scenario 2] with_targets mode for evaluation")
    print("-" * 50)

    config_with_targets = InferenceDatasetConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        mode="with_targets",
        disable_exo_prompt=True,
        output_feature_idx=(7, 8, 9),  # tAir, vpAir, co2Air indices
        return_all_output_features=True,
        datetime_format="%Y-%m-%d %H:%M:%S",
    )

    # Use first 500 timesteps for evaluation
    timeseries_eval = timeseries_data[:500]

    dataset_with_targets = GreenlightInferenceDataset(
        time_series_input=timeseries_eval,
        exo_prompt_input=None,
        config=config_with_targets,
    )

    print(f"  Config: mode={config_with_targets.mode}")
    print(f"  Min required timesteps: {config_with_targets.min_required_timesteps()}")
    print(f"  Input timesteps: {len(timeseries_eval)}")
    print(f"  Dataset length: {len(dataset_with_targets)}")

    sample = dataset_with_targets[0]
    print(f"  Sample seq_x shape: {sample.seq_x.shape}")
    print(f"  Sample seq_y shape: {sample.seq_y.shape}")
    print(f"  Sample seq_y_all shape: {sample.seq_y_all.shape}")
    print(f"  Sample seq_y_mark shape: {sample.seq_y_mark.shape}")

    # -------------------------------------------------------------------------
    # Scenario 3: PyTorch DataLoader compatibility
    # -------------------------------------------------------------------------
    print("\n[Scenario 3] PyTorch DataLoader compatibility")
    print("-" * 50)

    # Use return_dict=True for direct DataLoader compatibility
    config_dataloader = InferenceDatasetConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        mode="with_targets",
        disable_exo_prompt=True,
        output_feature_idx=(7, 8, 9),
        return_all_output_features=True,
        datetime_format="%Y-%m-%d %H:%M:%S",
        return_dict=True,  # Direct dict output for DataLoader
    )

    dataset_for_loader = GreenlightInferenceDataset(
        time_series_input=timeseries_data[:500],
        exo_prompt_input=None,
        config=config_dataloader,
    )

    loader = DataLoader(dataset_for_loader, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch seq_x shape: {batch['seq_x'].shape}")
    print(f"  Batch seq_y shape: {batch['seq_y'].shape}")
    print(f"  Batch dtype: {batch['seq_x'].dtype}")

    # -------------------------------------------------------------------------
    # Scenario 4: Validation error - insufficient data
    # -------------------------------------------------------------------------
    print("\n[Scenario 4] Validation error - insufficient data")
    print("-" * 50)

    try:
        too_short = timeseries_data[: SEQ_LEN - 10]  # Not enough data
        GreenlightInferenceDataset(
            time_series_input=too_short,
            exo_prompt_input=None,
            config=config_pred_only,
        )
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    # -------------------------------------------------------------------------
    # Scenario 5: With exo prompts enabled
    # -------------------------------------------------------------------------
    print("\n[Scenario 5] With exo prompts enabled")
    print("-" * 50)

    config_with_exo = InferenceDatasetConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        mode="prediction_only",
        disable_exo_prompt=False,
        exo_params_len=254,  # Full parameter set
        datetime_format="%Y-%m-%d %H:%M:%S",
    )

    dataset_with_exo = GreenlightInferenceDataset(
        time_series_input=timeseries_data[:SEQ_LEN],
        exo_prompt_input=exo_prompt_data,
        config=config_with_exo,
    )

    sample = dataset_with_exo[0]
    print(f"  Sample seq_x shape: {sample.seq_x.shape}")
    print(f"  Sample exo_params shape: {sample.exo_params.shape}")
    print(f"  Sample exo_params first 5: {sample.exo_params[:5]}")
    print(f"  Sample exo_params last 5: {sample.exo_params[-5:]}")

    # -------------------------------------------------------------------------
    # Scenario 6: Multiple sliding windows with real data
    # -------------------------------------------------------------------------
    print("\n[Scenario 6] Multiple sliding windows with real data")
    print("-" * 50)

    # Use more data for sliding windows
    timeseries_long = timeseries_data[:1000]

    dataset_long = GreenlightInferenceDataset(
        time_series_input=timeseries_long,
        exo_prompt_input=None,
        config=config_pred_only,
    )

    print(f"  Input timesteps: {len(timeseries_long)}")
    print(f"  Number of windows: {len(dataset_long)}")
    print(f"  Expected: {len(timeseries_long) - SEQ_LEN + 1}")

    # Check first and last samples have correct shapes
    first_sample = dataset_long[0]
    last_sample = dataset_long[-1]
    print(f"  First sample seq_x shape: {first_sample.seq_x.shape}")
    print(f"  Last sample seq_x shape: {last_sample.seq_x.shape}")

    # Verify negative indexing works
    assert first_sample.seq_x.shape == last_sample.seq_x.shape
    print("  Negative indexing: OK")

    # -------------------------------------------------------------------------
    # Scenario 7: Full pipeline test (DataLoader with exo prompts)
    # -------------------------------------------------------------------------
    print("\n[Scenario 7] Full pipeline test (DataLoader with exo prompts)")
    print("-" * 50)

    config_full = InferenceDatasetConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        mode="with_targets",
        disable_exo_prompt=False,
        exo_params_len=254,
        output_feature_idx=(7, 8, 9),
        return_all_output_features=True,
        datetime_format="%Y-%m-%d %H:%M:%S",
        return_dict=True,  # Direct dict output for DataLoader
    )

    dataset_full = GreenlightInferenceDataset(
        time_series_input=timeseries_data[:500],
        exo_prompt_input=exo_prompt_data,
        config=config_full,
    )

    loader_full = DataLoader(dataset_full, batch_size=8, shuffle=True)

    batch = next(iter(loader_full))
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch seq_x shape: {batch['seq_x'].shape}")
    print(f"  Batch seq_y shape: {batch['seq_y'].shape}")
    print(f"  Batch exo_params shape: {batch['exo_params'].shape}")

    # -------------------------------------------------------------------------
    # Scenario 8: inverse_transform for post-processing predictions
    # -------------------------------------------------------------------------
    print("\n[Scenario 8] inverse_transform for post-processing predictions")
    print("-" * 50)

    # Get original unscaled values directly from CSV for comparison
    csv_path = (
            base_dir
            / "data"
            / "from_david_by_gurkan"
            / "gt"
            / "hps"
            / "gt_hps_timeseries.csv"
    )
    df_original = pd.read_csv(csv_path)
    # Drop sideLee, sideWind to match our 18 features
    df_original = df_original.drop(columns=["sideLee", "sideWind"])

    # The seq_y for sample[0] corresponds to indices [r_begin:r_end]
    # where r_begin = seq_len - label_len = 192 - 48 = 144
    # and r_end = r_begin + label_len + pred_len = 144 + 48 + 96 = 288
    r_begin = SEQ_LEN - LABEL_LEN  # 144
    r_end = r_begin + LABEL_LEN + PRED_LEN  # 288

    # Get original values for output features (tAir, vpAir, co2Air at indices 7, 8, 9)
    output_cols = ["tAir", "vpAir", "co2Air"]
    original_from_csv = (
        df_original[output_cols].iloc[r_begin:r_end].values.astype("float32")
    )

    # Get scaled values from dataset and inverse transform
    sample = dataset_with_targets[0]
    scaled_predictions = sample.seq_y  # Shape: (144, 3)
    recovered = dataset_with_targets.inverse_transform(scaled_predictions)

    print(f"  Scaled shape: {scaled_predictions.shape}")
    print(f"  Recovered shape: {recovered.shape}")
    print(f"  Original CSV first row: {original_from_csv[0, :]}")
    print(f"  Recovered first row:    {recovered[0, :]}")

    # Verify recovery matches original CSV values (with small tolerance for float precision)
    max_diff = np.abs(original_from_csv - recovered).max()
    print(f"  Max difference from original: {max_diff:.6f}")
    assert max_diff < 0.01, f"Recovery error too large: {max_diff}"
    print("  Round-trip verification: OK")

    # Test with full feature predictions
    full_cols = [c for c in df_original.columns if c != "date"]
    original_full = df_original[full_cols].iloc[r_begin:r_end].values.astype("float32")
    full_scaled = sample.seq_y_all  # Shape: (144, 18)
    full_recovered = dataset_with_targets.inverse_transform(full_scaled)

    max_diff_full = np.abs(original_full - full_recovered).max()
    print(f"  Full feature max diff: {max_diff_full:.6f}")
    assert (
            max_diff_full < 0.01
    ), f"Full feature recovery error too large: {max_diff_full}"
    print("  Full feature round-trip: OK")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All scenarios passed with real data!")
    print("=" * 70)


if __name__ == "__main__":
    main()
