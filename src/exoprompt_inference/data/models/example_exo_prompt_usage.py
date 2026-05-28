"""
Example usage of ExoPromptInput for inference pipeline.
"""

import json
from pathlib import Path

import numpy as np

from exoprompt_inference.data.models.exo_prompt_input import ExoPromptInput


def example_1_from_json_file():
    """Load ExoPromptInput from JSON file (e.g., parameter file from calibration)."""
    print("=" * 60)
    print("Example 1: Loading from JSON file")
    print("=" * 60)

    # Path to example parameter file
    json_path = Path(__file__).parents[
                    2] / "data" / "from_david_by_gurkan" / "gt" / "hps" / "climate_model_hps_params.json"

    # Load JSON
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} parameters from JSON")
    print(f"Sample parameters: {list(raw_data.keys())[:5]}")

    # Remove lambdaShScrPer if present (matching dataset preprocessing)
    if "lambdaShScrPer" in raw_data:
        if raw_data["lambdaShScrPer"] is None:
            del raw_data["lambdaShScrPer"]
            print("Removed lambdaShScrPer (None value, not part of exoprompt)")

    # Create validated input
    exo_prompt = ExoPromptInput.from_dict(raw_data)
    print("\n✓ Validation passed!")
    print(f"\n{exo_prompt}")

    # Convert to numpy array for model inference
    param_array = exo_prompt.to_array()
    print(f"\nArray shape: {param_array.shape}")
    print(f"Array dtype: {param_array.dtype}")
    print(f"First 5 values: {param_array[:5]}")

    return exo_prompt


def example_2_array_conversion():
    """Demonstrate array conversion with sorted keys."""
    print("\n" + "=" * 60)
    print("Example 2: Array conversion and round-trip")
    print("=" * 60)

    # Load from file
    json_path = Path(__file__).parents[
                    2] / "data" / "from_david_by_gurkan" / "gt" / "hps" / "climate_model_hps_params.json"
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Remove lambdaShScrPer if present (matching dataset preprocessing)
    if "lambdaShScrPer" in raw_data and raw_data["lambdaShScrPer"] is None:
        del raw_data["lambdaShScrPer"]

    exo_prompt = ExoPromptInput.from_dict(raw_data)

    # Convert to array (sorted by key name)
    arr = exo_prompt.to_array()
    print(f"Original array shape: {arr.shape}")

    # Get parameter names in sorted order
    param_names = ExoPromptInput.get_feature_names()
    print(f"Number of parameters: {len(param_names)}")

    # Round trip: array → object → array
    exo_prompt2 = ExoPromptInput.from_array(arr, param_names)
    arr2 = exo_prompt2.to_array()

    assert np.allclose(arr, arr2)
    print("✓ Round-trip conversion preserves values")

    # Show parameter ordering
    print(f"\nFirst 10 parameters (sorted): {param_names[:10]}")
    print(f"Last 10 parameters (sorted): {param_names[-10:]}")


def example_3_validation():
    """Demonstrate automatic validation."""
    print("\n" + "=" * 60)
    print("Example 3: Automatic validation")
    print("=" * 60)

    # This will raise validation error (temperature too high)
    try:
        invalid_data = {
            "alfaLeafAir": 5.0,
            "L": 2.45e6,
            "sigma": 5.67e-8,
            "tSpDay": 100.0,  # ❌ Too high! Max is 40°C
            # ... would need all required fields
        }
        # Note: This will fail because not all required fields are provided
        ExoPromptInput.from_dict(invalid_data)
    except Exception as e:
        print(f"✓ Validation caught invalid/incomplete input:")
        print(f"  {type(e).__name__}: {str(e)[:150]}...")


def example_4_subset_parameters():
    """Demonstrate working with parameter subsets (like exo_params_to_take)."""
    print("\n" + "=" * 60)
    print("Example 4: Parameter subsets")
    print("=" * 60)

    # Load full parameters
    json_path = Path(__file__).parents[
                    2] / "data" / "from_david_by_gurkan" / "gt" / "hps" / "climate_model_hps_params.json"
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Remove lambdaShScrPer if present (matching dataset preprocessing)
    if "lambdaShScrPer" in raw_data and raw_data["lambdaShScrPer"] is None:
        del raw_data["lambdaShScrPer"]

    full_exo_prompt = ExoPromptInput.from_dict(raw_data)

    # Simulate filtering (like exo_params_to_take in dataset)
    params_to_keep = ["tSpDay", "tSpNight", "co2SpDay", "laiMax", "aFlr"]

    # Extract subset
    subset_data = {k: v for k, v in raw_data.items() if k in params_to_keep}
    print(f"Filtered to {len(subset_data)} parameters: {list(subset_data.keys())}")

    # Note: This would fail validation because not all required fields are present
    # In practice, you'd need to handle this differently (e.g., with Optional fields)
    # or keep all parameters but only use a subset for the model

    # Instead, demonstrate extracting subset from array
    full_array = full_exo_prompt.to_array()
    all_param_names = ExoPromptInput.get_feature_names()

    # Get indices of parameters we want
    subset_indices = [all_param_names.index(k) for k in params_to_keep if k in all_param_names]
    subset_array = full_array[subset_indices]

    print(f"Extracted subset array shape: {subset_array.shape}")
    print(f"Values: {subset_array}")


def example_5_partial_parameters():
    """Demonstrate creating ExoPromptInput with partial parameters (like exo_params_to_take)."""
    print("\n" + "=" * 60)
    print("Example 5: Partial parameter sets")
    print("=" * 60)

    # Create instance with only a few control parameters (like cLeakage experiments)
    partial_params = {
        "tSpDay": 19.5,
        "tSpNight": 16.5,
        "co2SpDay": 800.0,
        "laiMax": 3.0,
        "aFlr": 144.0,
        "cLeakage": 3e-5,  # The parameter being varied in experiments
    }

    exo_prompt = ExoPromptInput.from_dict(partial_params)
    print(f"\n{exo_prompt}")

    # Convert to array
    arr = exo_prompt.to_array()
    print(f"\nArray shape: {arr.shape}")  # Will be (6,) not (254,)
    print(f"Array values: {arr}")

    # Get present parameter names
    present_names = exo_prompt.get_present_param_names()
    print(f"\nPresent parameters: {present_names}")

    # Verify array length matches present parameters
    assert len(arr) == len(present_names) == 6
    print("✓ Array length matches number of present parameters")

    # This is useful for experiments like:
    # - c_leakage experiments (varying one parameter across datasets)
    # - exo_params_to_take (filtering to specific parameter subsets)
    print("\n✓ Partial parameter sets supported!")


if __name__ == "__main__":
    example_1_from_json_file()
    example_2_array_conversion()
    example_3_validation()
    example_4_subset_parameters()
    example_5_partial_parameters()

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)
