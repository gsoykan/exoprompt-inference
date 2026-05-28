"""
Loader for exogenous prompt parameters from JSON files.

This loader reads JSON files containing greenhouse model parameters and converts
them to ExoPromptInput instances for inference.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from exoprompt_inference.data.models import ExoPromptInput


class ExoPromptJsonLoader:
    """
    Loader for exogenous prompt parameters from JSON files.

    Reads JSON files containing greenhouse model parameters (e.g., from calibration,
    sensitivity analysis, or predefined configurations) and converts them to
    ExoPromptInput instances.

    Attributes:
        json_path: Path to the JSON file containing parameters
        discarded_features: Tuple of parameter names to discard during loading
        data: Loaded ExoPromptInput instance
    """

    def __init__(
            self,
            json_path: str | Path,
            discarded_features: Optional[Tuple[str, ...]] = ("lambdaShScrPer",),
    ):
        """
        Initialize the loader and load data from JSON.

        Args:
            json_path: Path to the JSON file containing exogenous parameters
            discarded_features: Tuple of parameter names to discard. Default
                              discards 'lambdaShScrPer' which is not part of
                              the exoprompt (can be None in some files).
        """
        self.json_path = Path(json_path)
        self.discarded_features = discarded_features
        self.data: ExoPromptInput = self._read_data()

    def _read_data(self) -> ExoPromptInput:
        """
        Read JSON file and convert to ExoPromptInput instance.

        Returns:
            ExoPromptInput instance with parameters from JSON
        """
        # Read JSON file
        with open(self.json_path, "r") as f:
            parameters_json: Dict[str, Any] = json.load(f)

        # Remove discarded features (if present and None)
        if self.discarded_features is not None:
            for discarded_feature in self.discarded_features:
                if discarded_feature in parameters_json:
                    # Only remove if None (lambdaShScrPer case)
                    # Otherwise keep it - user might want to use it
                    if parameters_json[discarded_feature] is None:
                        del parameters_json[discarded_feature]

        # Create ExoPromptInput using from_dict for consistency
        return ExoPromptInput.from_dict(parameters_json)

    def __str__(self) -> str:
        """Human-readable string representation."""
        num_params = len(self.data.to_dict())
        return (
            f"ExoPromptJsonLoader:\n"
            f"  JSON path: {self.json_path}\n"
            f"  Discarded features: {self.discarded_features}\n"
            f"  Parameters loaded: {num_params}/254\n"
            f"  Sample params: tSpDay={self.data.tSpDay}, "
            f"co2SpDay={self.data.co2SpDay}, laiMax={self.data.laiMax}"
        )


if __name__ == "__main__":
    # Example usage
    base_dir = os.getenv("BASE_DIR")
    if not base_dir:
        base_dir = Path(__file__).parents[3]  # Default to project root
        print(f"BASE_DIR not set, using: {base_dir}")

    # Load full parameter set
    json_path = (
            Path(base_dir)
            / "data"
            / "from_david_by_gurkan"
            / "gt"
            / "hps"
            / "climate_model_hps_params.json"
    )

    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        exit(1)

    # Load data
    loader = ExoPromptJsonLoader(json_path)
    print(loader)

    # Access the loaded ExoPromptInput
    print("\nAccessing loaded data:")
    print(f"  ExoPromptInput: {loader.data}")

    # Convert to array
    print("\nArray conversion:")
    arr = loader.data.to_array()
    print(f"  Array shape: {arr.shape}")
    print(f"  First 5 values: {arr[:5]}")

    # Get present parameter names
    print("\nParameter names:")
    param_names = loader.data.get_present_param_names()
    print(f"  Total: {len(param_names)}")
    print(f"  First 10: {param_names[:10]}")
    print(f"  Last 10: {param_names[-10:]}")
