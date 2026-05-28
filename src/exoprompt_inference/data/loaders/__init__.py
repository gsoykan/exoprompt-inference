"""
Data loaders for inference app.
"""

from exoprompt_inference.data.loaders.gt_csv_v1_loader import GTCSVV1Loader
from exoprompt_inference.data.loaders.exo_prompt_json_loader import ExoPromptJsonLoader

__all__ = ["GTCSVV1Loader", "ExoPromptJsonLoader"]
