"""
Example usage of BaseModelInput for inference pipeline.
"""

from datetime import datetime

import numpy as np

from exoprompt_inference.data.models.base_model_input import BaseModelInput


def example_1_from_dict():
    """Create BaseModelInput from dictionary (e.g., JSON API, CSV row)."""
    print("=" * 60)
    print("Example 1: Creating from dictionary")
    print("=" * 60)

    # Example: Data from JSON API or CSV
    raw_data = {
        'timestamp': datetime(2024, 6, 15, 14, 30, 0),
        'iGlob': 450.0,
        'tOut': 18.5,
        'vpOut': 1200.0,
        'co2Out': 400.0,
        'wind': 3.2,
        'tSky': 10.0,
        'tSoOut': 15.0,
        'tAir': 20.5,
        'vpAir': 1500.0,
        'co2Air': 800.0,
        'shScr': 0.3,
        'blScr': 0.0,
        'roof': 0.5,
        'tPipe': 35.0,
        'tGroPipe': 30.0,
        'lamp': 0.8,
        'intLamp': 0.6,
        'extCo2': 0.4,
    }

    # Create validated input
    model_input = BaseModelInput.from_dict(raw_data)
    print(model_input)
    print("\nValidation passed! ✓")

    # Convert to numpy array for model inference
    input_array = model_input.to_array()
    print(f"\nArray shape: {input_array.shape}")
    print(f"Array dtype: {input_array.dtype}")
    print(f"First 5 values: {input_array[:5]}")

    return model_input


def example_2_validation():
    """Demonstrate automatic validation."""
    print("\n" + "=" * 60)
    print("Example 2: Automatic validation")
    print("=" * 60)

    # This will raise validation error (temperature too high)
    try:
        invalid_data = {
            'timestamp': datetime(2024, 6, 15, 14, 30, 0),
            'iGlob': 450.0,
            'tOut': 150.0,  # ❌ Too high! Max is 50°C
            'vpOut': 1200.0,
            'co2Out': 400.0,
            'wind': 3.2,
            'tSky': 10.0,
            'tSoOut': 15.0,
            'tAir': 20.5,
            'vpAir': 1500.0,
            'co2Air': 800.0,
            'shScr': 0.3,
            'blScr': 0.0,
            'roof': 0.5,
            'tPipe': 35.0,
            'tGroPipe': 30.0,
            'lamp': 0.8,
            'intLamp': 0.6,
            'extCo2': 0.4,
        }
        BaseModelInput.from_dict(invalid_data)
    except Exception as e:
        print(f"✓ Validation caught invalid input:")
        print(f"  {type(e).__name__}: {e}")


def example_3_from_array():
    """Create BaseModelInput from numpy array (e.g., model output preprocessing)."""
    print("\n" + "=" * 60)
    print("Example 3: Creating from numpy array")
    print("=" * 60)

    # Example: Data from numpy array (maybe from a CSV or sensor buffer)
    arr = np.array([
        450.0,  # iGlob
        18.5,  # tOut
        1200.0,  # vpOut
        400.0,  # co2Out
        3.2,  # wind
        10.0,  # tSky
        15.0,  # tSoOut
        20.5,  # tAir
        1500.0,  # vpAir
        800.0,  # co2Air
        0.3,  # shScr
        0.0,  # blScr
        0.5,  # roof
        35.0,  # tPipe
        30.0,  # tGroPipe
        0.8,  # lamp
        0.6,  # intLamp
        0.4,  # extCo2
    ], dtype=np.float32)

    timestamp = datetime(2024, 6, 15, 14, 30, 0)
    model_input = BaseModelInput.from_array(arr, timestamp)
    print(model_input)
    print("\n✓ Converted from array successfully")

    # Round trip: array → object → array
    arr2 = model_input.to_array()
    assert np.allclose(arr, arr2)
    print("✓ Round-trip conversion preserves values")


def example_4_feature_names():
    """Get feature names for plotting, analysis, etc."""
    print("\n" + "=" * 60)
    print("Example 4: Feature names")
    print("=" * 60)

    names = BaseModelInput.get_feature_names()
    print(f"Total features: {len(names)}")
    print(f"Feature names: {names}")


if __name__ == "__main__":
    example_1_from_dict()
    example_2_validation()
    example_3_from_array()
    example_4_feature_names()

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)
