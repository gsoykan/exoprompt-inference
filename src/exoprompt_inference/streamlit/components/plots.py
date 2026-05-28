"""Plotting utilities for greenhouse climate predictions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Feature names and units for display
FEATURE_INFO = {
    0: {"name": "tAir", "label": "Air Temperature", "unit": "°C"},
    1: {"name": "vpAir", "label": "Vapor Pressure", "unit": "Pa"},
    2: {"name": "co2Air", "label": "CO2 Concentration", "unit": "ppm"},
    3: {"name": "rhAir", "label": "Relative Humidity", "unit": "%"},
}


def plot_predictions(
        history: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray | None = None,
        history_steps_to_show: int = 48,
        history_timestamps: pd.DatetimeIndex | None = None,
        prediction_timestamps: pd.DatetimeIndex | None = None,
) -> go.Figure:
    """
    Create a plotly figure with predictions for tAir, vpAir, co2Air, and rhAir.

    Args:
        history: Historical data [seq_len, 4] (unscaled, output features + RH)
        predictions: Model predictions [pred_len, 4] (unscaled, output features + RH)
        ground_truth: Optional ground truth [pred_len, 4] (unscaled, output features + RH)
        history_steps_to_show: Number of historical steps to display
        history_timestamps: Optional timestamps for history data
        prediction_timestamps: Optional timestamps for prediction data

    Returns:
        Plotly figure with 4 subplots (tAir, vpAir, co2Air, rhAir)
    """
    # Limit history to last N steps (and corresponding timestamps)
    if history.shape[0] > history_steps_to_show:
        history = history[-history_steps_to_show:]
        if history_timestamps is not None:
            history_timestamps = history_timestamps[-history_steps_to_show:]

    # Create x-axis values
    hist_len = history.shape[0]
    pred_len = predictions.shape[0]

    use_timestamps = history_timestamps is not None and prediction_timestamps is not None

    if use_timestamps:
        x_history = history_timestamps
        x_pred = prediction_timestamps[:pred_len]
        # Prediction boundary is at the first prediction timestamp (use iloc for positional indexing)
        prediction_boundary = prediction_timestamps[0] if isinstance(prediction_timestamps, (list, np.ndarray)) else \
            prediction_timestamps.iloc[0]
    else:
        x_history = np.arange(-hist_len, 0)
        x_pred = np.arange(0, pred_len)
        prediction_boundary = 0

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"{FEATURE_INFO[i]['label']} ({FEATURE_INFO[i]['unit']})"
            for i in range(4)
        ],
    )

    colors = {
        "history": "#6c757d",  # gray
        "prediction": "#0d6efd",  # blue
        "ground_truth": "#198754",  # green
    }

    for i in range(4):
        row = i + 1

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=x_history,
                y=history[:, i],
                mode="lines",
                name="History" if i == 0 else None,
                line=dict(color=colors["history"], width=1.5),
                legendgroup="history",
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Predictions
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=predictions[:, i],
                mode="lines",
                name="Prediction" if i == 0 else None,
                line=dict(color=colors["prediction"], width=2),
                legendgroup="prediction",
                showlegend=(i == 0),
            ),
            row=row,
            col=1,
        )

        # Ground truth (if available)
        if ground_truth is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_pred,
                    y=ground_truth[:, i],
                    mode="lines",
                    name="Ground Truth" if i == 0 else None,
                    line=dict(color=colors["ground_truth"], width=2, dash="dash"),
                    legendgroup="ground_truth",
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )

        # Add vertical line at prediction boundary
        fig.add_vline(
            x=prediction_boundary,
            line_dash="dot",
            line_color="rgba(0,0,0,0.3)",
            row=row,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=800,  # Increased for 4 subplots
        title_text="Greenhouse Climate Predictions",
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )

    # Update x-axis label for bottom plot only
    x_axis_label = "Date/Time" if use_timestamps else "Timesteps (5 min intervals)"
    fig.update_xaxes(title_text=x_axis_label, row=4, col=1)

    return fig


def compute_metrics(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute MSE, MAE, and RRMSE for each output feature.

    Args:
        predictions: Model predictions [pred_len, 4]
        ground_truth: Ground truth values [pred_len, 4]

    Returns:
        Dict with metrics per feature. Each feature has:
            - MSE: Mean Squared Error
            - MAE: Mean Absolute Error
            - RRMSE: Relative Root Mean Squared Error (%)
            - unit: Unit of measurement
    """
    metrics = {}

    for i in range(4):
        pred = predictions[:, i]
        true = ground_truth[:, i]

        # Calculate metrics
        mse = float(np.mean((pred - true) ** 2))
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(mse))

        # Calculate RRMSE (Relative RMSE as percentage)
        # RRMSE = (RMSE / mean(observed)) * 100
        mean_true = float(np.mean(true))
        rrmse = (rmse / mean_true * 100) if mean_true != 0 else float('inf')

        feature_name = FEATURE_INFO[i]["name"]
        metrics[feature_name] = {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "RRMSE": rrmse,
            "unit": FEATURE_INFO[i]["unit"],
        }

    return metrics
