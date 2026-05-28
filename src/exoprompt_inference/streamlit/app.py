"""
Greenhouse Climate Predictor - Streamlit App

A single-page application for running inference with trained greenhouse
climate models.

Run with:
    streamlit run inference_app/streamlit/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download

from exoprompt_inference.streamlit.components.utils import add_rh_to_unscaled

# =============================================================================
# Hugging Face checkpoint registry
# =============================================================================

HF_REPO_ID = "gsoykan/exoprompt-checkpoints"

# Friendly label → filename inside HF_REPO_ID. Keep `200k_exo_hps` (the default)
# first so it's the natural pick for first-time visitors.
HF_MODELS: dict[str, str] = {
    "ExoPrompt — HPS (200k)": "200k_exo_hps.ckpt",
    "ExoPrompt — LED (200k)": "200k_exo_led.ckpt",
    "ExoPrompt — Mixed (200k)": "200k_exo_mixed.ckpt",
    "Vanilla Transformer — HPS (200k)": "200k_vanilla_hps.ckpt",
    "Vanilla Transformer — LED (200k)": "200k_vanilla_led.ckpt",
    "Vanilla Transformer — Mixed (200k)": "200k_vanilla_mixed.ckpt",
}

# =============================================================================
# Bundled sample data
# =============================================================================

# Repo root: src/exoprompt_inference/streamlit/app.py → parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]

# Friendly label → (csv path, json path) inside `examples/`.
SAMPLE_DATA: dict[str, dict[str, Path]] = {
    "HPS lighting (paper data)": {
        "csv": REPO_ROOT / "examples" / "hps" / "timeseries.csv",
        "json": REPO_ROOT / "examples" / "hps" / "exo_params.json",
    },
    "LED lighting (paper data)": {
        "csv": REPO_ROOT / "examples" / "led" / "timeseries.csv",
        "json": REPO_ROOT / "examples" / "led" / "exo_params.json",
    },
}

# Add project root to path for imports
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from exoprompt_inference.data.datasets.greenlight_inference_dataset import (
    GreenlightInferenceDataset,
    InferenceDatasetConfig,
)
from exoprompt_inference.data.loaders.exo_prompt_json_loader import ExoPromptJsonLoader
from exoprompt_inference.data.loaders.gt_csv_v1_loader import GTCSVV1Loader
from exoprompt_inference.inference.model_loader import (
    ModelType,
    load_model_from_checkpoint,
)
from exoprompt_inference.inference.predictor import ModelInputMode, Predictor
from exoprompt_inference.streamlit.components.plots import compute_metrics, plot_predictions

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Greenhouse Climate Predictor",
    page_icon="🌿",
    layout="wide",
)


# =============================================================================
# Cached Functions
# =============================================================================


@st.cache_resource
def cached_load_model(checkpoint_path: str, model_type: str, device: str):
    """Load model with caching to avoid reloading."""
    model_type_enum = ModelType(model_type)
    return load_model_from_checkpoint(checkpoint_path, model_type_enum, device)


@st.cache_resource(show_spinner="Downloading checkpoint from Hugging Face Hub…")
def cached_hf_download(repo_id: str, filename: str) -> str:
    """Fetch a checkpoint from the Hub, returning the cached local path."""
    return hf_hub_download(repo_id=repo_id, filename=filename)


@st.cache_data
def cached_load_csv(csv_content: bytes):
    """Load and parse CSV file."""
    # Save to temp file and load with GTCSVV1Loader
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    loader = GTCSVV1Loader(temp_path)
    return loader.data


@st.cache_data
def cached_load_json(json_content: bytes):
    """Load and parse JSON exo params file."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json_content)
        temp_path = f.name

    loader = ExoPromptJsonLoader(temp_path)
    return loader.data


def save_checkpoint_to_temp(checkpoint_content: bytes) -> str:
    """Save uploaded checkpoint to temporary file and return path."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        f.write(checkpoint_content)
        temp_path = f.name

    return temp_path


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar():
    """Render sidebar with data and model controls."""

    st.sidebar.title("🌿 Greenhouse Climate Predictor")
    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Data Section
    # -------------------------------------------------------------------------
    st.sidebar.header("📁 Data")

    # Sample-vs-upload toggle. Sample is the default so the demo is usable on
    # first visit without uploading anything.
    data_source = st.sidebar.radio(
        "Data Source",
        options=["Sample (paper data)", "Upload my own"],
        help="Use a bundled greenhouse trajectory from the paper, or upload your own.",
    )

    csv_bytes: bytes | None = None
    json_bytes: bytes | None = None

    if data_source == "Sample (paper data)":
        sample_label = st.sidebar.selectbox(
            "Scenario",
            options=list(SAMPLE_DATA.keys()),
            help="Pre-loaded greenhouse trajectories from the paper datasets.",
        )
        sample = SAMPLE_DATA[sample_label]
        csv_bytes = sample["csv"].read_bytes()
        json_bytes = sample["json"].read_bytes()
        st.sidebar.caption(
            f"Loaded `{sample['csv'].relative_to(REPO_ROOT)}` and "
            f"`{sample['json'].relative_to(REPO_ROOT)}`."
        )
    else:
        csv_file = st.sidebar.file_uploader(
            "Time Series CSV (required)",
            type=["csv"],
            help="Upload a CSV file with greenhouse time series data (18 features)",
        )
        json_file = st.sidebar.file_uploader(
            "Exo Params JSON (optional)",
            type=["json"],
            help="Upload a JSON file with greenhouse model parameters (254 params)",
        )
        if csv_file is not None:
            csv_bytes = csv_file.getvalue()
        if json_file is not None:
            json_bytes = json_file.getvalue()

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Model Section
    # -------------------------------------------------------------------------
    st.sidebar.header("🤖 Model")

    # Only Transformer-backed models are exposed in this demo.
    model_type = ModelType.TIME_SERIES_LIB_MODEL.value

    # Checkpoint selection method — HF Hub is the default so first-time
    # visitors get a working model without uploading anything.
    checkpoint_method = st.sidebar.radio(
        "Checkpoint Source",
        options=["Hugging Face Hub", "Upload File", "Local Path (dev only)"],
        help="Where to fetch the .ckpt file from",
    )

    checkpoint_file = None
    checkpoint_path_str = None
    hf_model_label = None

    if checkpoint_method == "Hugging Face Hub":
        hf_model_label = st.sidebar.selectbox(
            "Paper checkpoint",
            options=list(HF_MODELS.keys()),
            help=(
                f"Downloaded once from [{HF_REPO_ID}]"
                f"(https://huggingface.co/{HF_REPO_ID}) and cached locally."
            ),
        )
    elif checkpoint_method == "Upload File":
        # Get max upload size from config (default 200MB)
        max_size_mb = st.config.get_option("server.maxUploadSize") or 200

        checkpoint_file = st.sidebar.file_uploader(
            "Upload Checkpoint",
            type=["ckpt"],
            help=f"Upload a .ckpt checkpoint file (max {max_size_mb}MB). "
                 f"Increase limit in .streamlit/config.toml if needed.",
        )
    else:
        checkpoint_path_str = st.sidebar.text_input(
            "Checkpoint Path",
            value="",
            help="⚠️ Only works when running locally! "
                 "Full path to .ckpt file on the machine running this app.",
        )

    # Warning about alignment
    st.sidebar.warning(
        "⚠️ **User Responsibility**: Ensure your data and model are aligned "
        "(same features, scaling, seq_len, etc.)"
    )

    st.sidebar.markdown("---")

    # -------------------------------------------------------------------------
    # Settings Section
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Settings")

    # Device selection
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    if torch.backends.mps.is_available():
        available_devices.append("mps")

    device = st.sidebar.selectbox(
        "Device",
        options=available_devices,
        index=len(available_devices) - 1,  # Default to last (best available)
    )

    # Inference mode
    mode = st.sidebar.selectbox(
        "Mode",
        options=["prediction_only", "with_targets"],
        help="'with_targets' shows ground truth comparison if available in data",
    )

    # Display sequence parameters from session state (after model is loaded)
    if "model_params" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📐 Model Parameters")
        params = st.session_state["model_params"]
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.metric("Seq", params["seq_len"])
        with col2:
            st.metric("Label", params["label_len"])
        with col3:
            st.metric("Pred", params["pred_len"])
        if params.get("uses_exo_prompt"):
            st.sidebar.success("✓ Uses Exo Prompt")
        else:
            st.sidebar.info("○ No Exo Prompt")
    else:
        st.sidebar.info(
            "ℹ️ Sequence parameters will be shown after loading the model."
        )

    return {
        "csv_bytes": csv_bytes,
        "json_bytes": json_bytes,
        "model_type": model_type,
        "checkpoint_method": checkpoint_method,
        "checkpoint_file": checkpoint_file,
        "checkpoint_path_str": checkpoint_path_str,
        "hf_model_label": hf_model_label,
        "device": device,
        "mode": mode,
    }


# =============================================================================
# Main Content
# =============================================================================


def render_main(config: dict):
    """Render main content area."""

    st.title("🌿 Greenhouse Climate Predictor")
    st.markdown(
        "Predict indoor climate (temperature, vapor pressure, CO₂) with the "
        "**[ExoPrompt](https://doi.org/10.1016/j.compag.2026.111673)** transformer "
        "— vanilla or conditioned on 254 structural / environmental / crop parameters. "
        "Relative humidity is derived from temperature and vapor pressure."
    )
    st.markdown(
        "📄 [**Paper**](https://doi.org/10.1016/j.compag.2026.111673)"
        " &nbsp;·&nbsp; 💻 [**Original paper repo**](https://github.com/gsoykan/ExoPrompt)"
        " &nbsp;·&nbsp; 🛠️ [**Demo source**](https://github.com/gsoykan/exoprompt-inference)"
        " &nbsp;·&nbsp; 🤗 [**Checkpoints**](https://huggingface.co/gsoykan/exoprompt-checkpoints)",
        unsafe_allow_html=True,
    )
    st.divider()

    # Check if we have required inputs
    if config["csv_bytes"] is None:
        st.info("👈 Please upload a CSV file in the sidebar to get started.")
        return

    # Check checkpoint based on method
    if config["checkpoint_method"] == "Hugging Face Hub":
        if not config["hf_model_label"]:
            st.info("👈 Please select a paper checkpoint in the sidebar.")
            return
    elif config["checkpoint_method"] == "Upload File":
        if config["checkpoint_file"] is None:
            st.info("👈 Please upload a checkpoint file in the sidebar.")
            return
    else:  # Local Path (dev only)
        if not config["checkpoint_path_str"]:
            st.info("👈 Please provide a checkpoint path in the sidebar.")
            return

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    with st.spinner("Loading data..."):  # type: ignore
        try:
            timeseries_data = cached_load_csv(config["csv_bytes"])
            st.success(f"Loaded {len(timeseries_data)} timesteps from CSV")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            return

        exo_prompt_data = None
        if config["json_bytes"] is not None:
            try:
                exo_prompt_data = cached_load_json(config["json_bytes"])
                st.success(
                    f"Loaded {len(exo_prompt_data.get_present_param_names())} exo params"
                )
            except Exception as e:
                st.error(f"Failed to load JSON: {e}")
                return

    # -------------------------------------------------------------------------
    # Load Model & Predict
    # -------------------------------------------------------------------------
    st.markdown("---")
    predict_button = st.button("🔮 Load Model & Dataset", type="primary", use_container_width=True)

    if predict_button:
        # Determine checkpoint path based on method
        if config["checkpoint_method"] == "Hugging Face Hub":
            filename = HF_MODELS[config["hf_model_label"]]
            checkpoint_path = cached_hf_download(HF_REPO_ID, filename)
        elif config["checkpoint_method"] == "Upload File":
            checkpoint_path = save_checkpoint_to_temp(config["checkpoint_file"].getvalue())
        else:  # Local Path (dev only)
            checkpoint_path = config["checkpoint_path_str"]

        # Load model
        with st.spinner("Loading model..."):  # type: ignore
            try:
                model = cached_load_model(
                    checkpoint_path,
                    config["model_type"],
                    config["device"],
                )
                st.success(f"Model loaded: {type(model).__name__}")
            except FileNotFoundError:
                st.error(f"Checkpoint not found: {checkpoint_path}")
                return
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

        # Determine model input mode from model config
        model_configs = model.model_configs
        enable_exo_prompt = getattr(model_configs, "enable_exo_prompt_tuning", False)
        if enable_exo_prompt:
            exo_prompt_dim = getattr(model_configs, "exo_prompt_dim", 254)
            if exo_prompt_dim == 1:
                model_input_mode = ModelInputMode.EXO_PROMPT_C_LEAKAGE
            else:
                model_input_mode = ModelInputMode.EXO_PROMPT_254
        else:
            model_input_mode = ModelInputMode.VANILLA

        # Create predictor
        predictor = Predictor(model=model, model_input_mode=model_input_mode)

        # Extract sequence parameters from predictor and store in session state
        seq_len = predictor.model.model_configs.seq_len
        label_len = predictor.label_len
        pred_len = predictor.pred_len

        st.session_state["model_params"] = {
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "uses_exo_prompt": predictor.uses_exo_prompt,
        }

        # Create dataset with parameters from model
        with st.spinner("Creating dataset..."):  # type: ignore
            try:
                dataset_config = InferenceDatasetConfig(
                    seq_len=seq_len,
                    label_len=label_len,
                    pred_len=pred_len,
                    mode=config["mode"],
                    disable_exo_prompt=(exo_prompt_data is None),
                    exo_params_len=254,
                    output_feature_idx=(7, 8, 9),  # tAir, vpAir, co2Air
                    return_all_output_features=True,
                    datetime_format="%Y-%m-%d %H:%M:%S",
                    return_dict=False,
                )

                dataset = GreenlightInferenceDataset(
                    time_series_input=timeseries_data,
                    exo_prompt_input=exo_prompt_data,
                    config=dataset_config,
                )

                st.success(f"Dataset created with {len(dataset)} samples")

                # Store in session state
                st.session_state["predictor"] = predictor
                st.session_state["dataset"] = dataset
                st.session_state["pred_len"] = pred_len
                st.session_state["mode"] = config["mode"]

            except Exception as e:
                st.error(f"Failed to create dataset: {e}")
                return

    # -------------------------------------------------------------------------
    # Sample Selection & Prediction (outside button block)
    # -------------------------------------------------------------------------
    # Only show if model and dataset are loaded
    if "predictor" in st.session_state and "dataset" in st.session_state:
        predictor = st.session_state["predictor"]
        dataset = st.session_state["dataset"]
        pred_len = st.session_state["pred_len"]
        mode = st.session_state["mode"]

        # Sample selection
        st.markdown("---")
        st.subheader("📊 Sample Selection")

        sample_idx = st.slider(
            "Select Sample Index",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            key="sample_selector",
        )

        # Get sample and timestamps
        sample = dataset[sample_idx]
        timestamps_for_sample = dataset.get_timestamps_for_sample(sample_idx)

        # Prepare tensors
        seq_x = torch.from_numpy(sample.seq_x).unsqueeze(0).float()
        seq_x_mark = torch.from_numpy(sample.seq_x_mark).unsqueeze(0).float()
        seq_y_mark = torch.from_numpy(sample.seq_y_mark).unsqueeze(0).float()

        exo_prompt = None
        if sample.exo_params is not None and predictor.uses_exo_prompt:
            exo_prompt = torch.from_numpy(sample.exo_params).unsqueeze(0).float()

        # Run prediction
        with st.spinner("Running prediction..."):  # type: ignore
            try:
                output = predictor.predict(
                    seq_x=seq_x,
                    seq_x_mark=seq_x_mark,
                    seq_y_mark=seq_y_mark,
                    exo_prompt=exo_prompt,
                )

                # Convert to numpy and inverse transform
                output_np = output.cpu().numpy()[0]  # Remove batch dim
                predictions_unscaled = dataset.inverse_transform(
                    output_np.reshape(1, -1, 3)
                )[0]
                predictions_unscaled = add_rh_to_unscaled(predictions_unscaled)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        # -------------------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Predictions")

        # Get history for plotting (output features only: tAir, vpAir, co2Air)
        # These are at indices 7, 8, 9 in the full feature set
        history_scaled = sample.seq_x[:, [7, 8, 9]]
        history_unscaled = dataset.inverse_transform(
            history_scaled.reshape(1, -1, 3)
        )[0]
        history_unscaled = add_rh_to_unscaled(history_unscaled)

        # Get ground truth if available
        ground_truth_unscaled = None
        if mode == "with_targets" and sample.seq_y is not None:
            # seq_y contains label_len + pred_len, we want only pred_len
            gt_scaled = sample.seq_y[-pred_len:]
            ground_truth_unscaled = dataset.inverse_transform(
                gt_scaled.reshape(1, -1, 3)
            )[0]
            ground_truth_unscaled = add_rh_to_unscaled(ground_truth_unscaled)

        # Create plot with timestamps
        fig = plot_predictions(
            history=history_unscaled,
            predictions=predictions_unscaled,
            ground_truth=ground_truth_unscaled,
            history_steps_to_show=192,
            history_timestamps=timestamps_for_sample['input'],
            prediction_timestamps=timestamps_for_sample['prediction'],
        )
        st.plotly_chart(fig, width='stretch')

        # -------------------------------------------------------------------------
        # Metrics (if ground truth available)
        # -------------------------------------------------------------------------
        if ground_truth_unscaled is not None:
            st.markdown("---")
            st.subheader("📊 Metrics")

            metrics = compute_metrics(predictions_unscaled, ground_truth_unscaled)

            cols = st.columns(4)
            for i, (feature, values) in enumerate(metrics.items()):
                with cols[i]:
                    st.markdown(f"**{feature}** ({values['unit']})")
                    st.metric("MSE", f"{values['MSE']:.4f}")
                    st.metric("RMSE", f"{values['RMSE']:.4f}")
                    st.metric("MAE", f"{values['MAE']:.4f}")
                    st.metric("RRMSE", f"{values['RRMSE']:.2f}%")

        # -------------------------------------------------------------------------
        # Download
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📥 Download Predictions")

        # Create DataFrame for download with timestamps
        df_predictions = pd.DataFrame(
            predictions_unscaled,
            columns=["tAir (°C)", "vpAir (Pa)", "co2Air (ppm)", "rhAir (%)"],
            index=timestamps_for_sample['prediction'],
        )
        df_predictions.index.name = "timestamp"

        csv_data = df_predictions.to_csv()
        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("👆 Click 'Load Model & Dataset' button to start making predictions.")


# =============================================================================
# Main
# =============================================================================


def main():
    config = render_sidebar()
    render_main(config)


if __name__ == "__main__":
    main()
