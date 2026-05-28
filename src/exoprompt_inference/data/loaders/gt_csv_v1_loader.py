"""
Loader for greenhouse ground truth CSV data (Version 1 format).

This loader reads CSV files containing greenhouse time series data and converts
them to BaseModelInput instances for inference.

Version 1 format:
- CSV with 'date' column as first column
- 18 or 20 features (depending on presence of sideLee/sideWind)
- Compatible with original ground truth data format
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from exoprompt_inference.data.models.base_model_input import BaseModelInput

load_dotenv()


class GTCSVV1Loader:
    """
    Loader for greenhouse ground truth CSV data (Version 1 format).

    Reads CSV files containing time series data with greenhouse environmental
    conditions and converts each row to a BaseModelInput instance.

    Expected CSV format (Version 1):
    - First column: 'date' (timestamp)
    - Remaining columns: 18 features matching BaseModelInput fields
    - Optional columns 'sideLee' and 'sideWind' are discarded by default
      (these exist in some ground truth data but not in simulation data)

    Attributes:
        csv_path: Path to the CSV file
        discarded_features: Tuple of feature names to discard during loading
        data: List of loaded BaseModelInput instances
    """

    def __init__(
        self,
        csv_path: str | Path,
        discarded_features: Optional[Tuple[str, ...]] = ("sideLee", "sideWind"),
    ):
        """
        Initialize the loader and load data from CSV.

        Args:
            csv_path: Path to the CSV file to load
            discarded_features: Tuple of column names to discard (e.g., features
                               not present in all datasets). Default discards
                               'sideLee' and 'sideWind' which exist in ground truth
                               but not in simulation data.
        """
        self.csv_path = Path(csv_path)
        self.discarded_features = discarded_features
        self.data: List[BaseModelInput] = self._read_data()

    def _read_data(self) -> List[BaseModelInput]:
        """
        Read CSV file and convert to BaseModelInput instances.

        Returns:
            List of BaseModelInput instances, one per row in the CSV
        """
        # Read CSV
        df = pd.read_csv(self.csv_path)

        # Discard unwanted features
        if self.discarded_features is not None:
            columns_to_drop = [col for col in self.discarded_features if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)

        # Ensure 'date' is first column
        cols = list(df.columns)
        if "date" not in cols:
            raise ValueError("CSV must contain 'date' column")
        cols.remove("date")
        df = df[["date"] + cols]

        # Extract data columns (everything except 'date')
        cols_data = df.columns[1:]
        df_data = df[cols_data]
        df_data = df_data.astype("float32")

        # Parse timestamps
        timestamps = pd.to_datetime(df["date"]).tolist()

        # Convert to BaseModelInput instances
        records = df_data.to_dict("records")
        model_inputs = [
            BaseModelInput.from_dict(data={"timestamp": t, **r})
            for t, r in zip(timestamps, records)
        ]

        return model_inputs

    def __len__(self) -> int:
        """Return number of loaded instances."""
        return len(self.data)

    def __getitem__(self, idx: int) -> BaseModelInput:
        """Get BaseModelInput instance by index."""
        return self.data[idx]

    def __iter__(self):
        """Iterate over BaseModelInput instances."""
        return iter(self.data)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"GTCSVV1Loader:\n"
            f"  CSV path: {self.csv_path}\n"
            f"  Discarded features: {self.discarded_features}\n"
            f"  Loaded instances: {len(self.data)}\n"
            f"  Time range: {self.data[0].timestamp if self.data else 'N/A'} → "
            f"{self.data[-1].timestamp if self.data else 'N/A'}"
        )


if __name__ == "__main__":
    # Example usage
    base_dir = os.getenv("BASE_DIR")
    if not base_dir:
        base_dir = Path(__file__).parents[3]  # Default to project root
        print(f"BASE_DIR not set, using: {base_dir}")

    gt_hps_time_series_csv_path = Path(base_dir) / "data" / "from_david_by_gurkan" / "gt" / "hps" / "gt_hps_timeseries.csv"

    if not gt_hps_time_series_csv_path.exists():
        print(f"CSV file not found: {gt_hps_time_series_csv_path}")
        exit(1)

    # Load data
    loader = GTCSVV1Loader(csv_path=gt_hps_time_series_csv_path)
    print(loader)

    # Show first few instances
    print("\nFirst 3 instances:")
    for i in range(min(3, len(loader))):
        print(f"\n[{i}] {loader[i]}")

    # Show array conversion
    print("\nFirst instance as array:")
    arr = loader[0].to_array()
    print(f"  Shape: {arr.shape}")
    print(f"  First 5 values: {arr[:5]}")
