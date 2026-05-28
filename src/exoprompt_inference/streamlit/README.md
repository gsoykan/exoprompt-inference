# Greenhouse Climate Predictor - Streamlit App

A single-page application for running inference with trained greenhouse climate models.

## Running Locally

```bash
# From project root
streamlit run inference_app/streamlit/app.py

# Or from this directory
streamlit run app.py
```

## Checkpoint File Size Limits

### Local Development
- **Upload File**: Limited by `.streamlit/config.toml` (default: 1GB)
- **Local Path**: No limit - provide path to checkpoint on your filesystem

### Deployment
- **Upload File**: Works on all platforms (size limit configurable)
- **Local Path**: ⚠️ Only works if checkpoint is on the server's filesystem

## Increasing Upload Size Limit

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 2000  # Size in MB (e.g., 2GB)
```

**Platform-specific notes:**
- **Streamlit Cloud**: Supports custom config.toml, but may have platform limits
- **Heroku**: Has slug size limits (~500MB), use cloud storage for large files
- **AWS/GCP/Azure**: Can configure as needed

## Deployment Options

### Option 1: Streamlit Cloud (Easiest)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your repository
4. The `.streamlit/config.toml` will be used automatically

### Option 2: Docker (Most Flexible)
```bash
# Create Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "inference_app/streamlit/app.py"]
```

### Option 3: Cloud Storage Integration (For Large Checkpoints)
For very large checkpoints (>1GB), consider loading from cloud storage instead of uploading:

```python
# Example: Load from S3
import boto3

s3 = boto3.client('s3')
s3.download_file('my-bucket', 'checkpoints/model.ckpt', '/tmp/model.ckpt')
model = load_model_from_checkpoint('/tmp/model.ckpt', ...)
```

## Features

- Upload CSV time series data (18 features)
- Optional JSON exogenous parameters (254 params)
- Model type selection (TimeSeriesLib or Custom)
- Device selection (CPU/CUDA/MPS)
- Interactive predictions visualization
- Metrics computation (MSE, MAE)
- Download predictions as CSV

## User Responsibilities

⚠️ Users must ensure:
- Data and model alignment (same features, scaling, sequence lengths)
- Correct model type selection
- Compatible checkpoint file
