---
title: ExoPrompt Greenhouse Climate Demo
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
tags:
  - streamlit
  - time-series-forecasting
  - greenhouse
  - exoprompt
pinned: false
license: mit
short_description: ExoPrompt paper demo — transformer-based greenhouse climate forecasting
---

# ExoPrompt — Greenhouse Climate Forecasting Demo

Interactive Streamlit demo for [ExoPrompt](https://doi.org/10.1016/j.compag.2026.111673):
transformer-based greenhouse climate forecasting with structured exogenous prompts and
physics-based simulation.

Two models are exposed:

- **ExoPrompt Transformer** — Transformer backbone conditioned on 254 structural,
  environmental, and crop parameters via learnable soft prompts.
- **Vanilla Transformer** — same Transformer backbone, no exogenous conditioning
  (baseline for comparison).

Both models are loaded from the published paper checkpoints (200k-sample HPS pretraining).

## Local development

```bash
# install deps
uv sync

# run streamlit locally
uv run streamlit run app.py
```

## Checkpoints

Checkpoints are hosted in a sibling Hugging Face model repo and downloaded on first
launch via `huggingface_hub.hf_hub_download` (cached on the Space's persistent disk).

## Project structure

```
exoprompt-inference/
├── app.py                              ← Streamlit entry (loaded by HF Spaces)
├── pyproject.toml
└── src/exoprompt_inference/
    ├── data/                           ← inference dataset + loaders
    ├── inference/                      ← model loader + Predictor
    ├── streamlit/                      ← Streamlit app + UI components
    └── _vendor/                        ← carved-out training code
        ├── models/                     ← lit module, abstract base, physi-net wrapper
        ├── utils/                      ← scaler, time features, pickle helper
        └── time_series_library/        ← Transformer backbone (only)
```

## Citation

```bibtex
@article{SOYKAN2026111673,
  title   = {ExoPrompt: Transformer-based greenhouse climate forecasting with structured conditioning and physics-based simulation},
  journal = {Computers and Electronics in Agriculture},
  volume  = {246},
  pages   = {111673},
  year    = {2026},
  author  = {Soykan, G{\"u}rkan and Babur, {\"O}nder and Liu, Qingzhi and Tekinerdogan, Bedir}
}
```
