"""
Base model input for greenhouse climate prediction.

This module defines the standardized input format for the greenhouse model.
All data sources (CSV, JSON, live sensors, etc.) should be converted to this format
before being used for inference.
"""

from datetime import datetime
from typing import Dict, Any, List

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class BaseModelInput(BaseModel):
    """
    Standardized input representation for greenhouse climate model.

    Contains 18 input features representing greenhouse environmental conditions,
    control actuators, and external weather conditions, plus a timestamp.

    Design Philosophy:
    - Timestamp is metadata (for sequence ordering, time feature extraction later)
    - to_array() returns only the 18 data features (no timestamp)
    - to_dict() returns all fields including timestamp
    - Time feature extraction happens in preprocessing layer, not here

    All temperature values are in Celsius [°C].
    All vapor pressure values are in Pascals [Pa].
    All CO2 values are in ppm.
    All control signals are normalized [0-1].
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # ========== Timestamp ==========

    timestamp: datetime = Field(
        description="Timestamp of the measurement"
    )

    # ========== External Weather Conditions (7 features) ==========

    iGlob: float = Field(
        description="Global solar radiation outside greenhouse [W/m²]",
        ge=0.0,
        le=1500.0,
    )

    tOut: float = Field(
        description="Outside air temperature [°C]",
        ge=-30.0,
        le=50.0,
    )

    vpOut: float = Field(
        description="Outside vapor pressure [Pa]",
        ge=0.0,
        le=5000.0,
    )

    co2Out: float = Field(
        description="Outside CO2 concentration [ppm]",
        ge=0.0,
        le=5000.0,
    )

    wind: float = Field(
        description="Wind speed [m/s]",
        ge=0.0,
        le=50.0,
    )

    tSky: float = Field(
        description="Sky temperature [°C]",
        ge=-50.0,
        le=30.0,
    )

    tSoOut: float = Field(
        description="Outside soil temperature [°C]",
        ge=-20.0,
        le=40.0,
    )

    # ========== Indoor Climate State (3 features) ==========

    tAir: float = Field(
        description="Indoor air temperature [°C]",
        ge=0.0,
        le=50.0,
    )

    vpAir: float = Field(
        description="Indoor vapor pressure [Pa]",
        ge=0.0,
        le=6000.0,
    )

    co2Air: float = Field(
        description="Indoor CO2 concentration [ppm]",
        ge=0.0,
        le=5000.0,
    )

    # ========== Control Actuators (8 features) ==========

    shScr: float = Field(
        description="Shade screen closure [0-1, 0=open, 1=closed]",
        ge=0.0,
        le=1.0,
    )

    blScr: float = Field(
        description="Blackout screen closure [0-1, 0=open, 1=closed]",
        ge=0.0,
        le=1.0,
    )

    roof: float = Field(
        description="Roof ventilation opening [0-1, 0=closed, 1=open]",
        ge=0.0,
        le=1.0,
    )

    tPipe: float = Field(
        description="Heating pipe temperature [°C]",
        ge=0.0,
        le=90.0,
    )

    tGroPipe: float = Field(
        description="Grow pipe temperature [°C]",
        ge=0.0,
        le=90.0,
    )

    lamp: float = Field(
        description="Lamp intensity [0-1, 0=off, 1=full power]",
        ge=0.0,
        le=1.0,
    )

    intLamp: float = Field(
        description="Internal lamp intensity [0-1, 0=off, 1=full power]",
        ge=0.0,
        le=1.0,
    )

    extCo2: float = Field(
        description="External CO2 dosing [0-1, 0=off, 1=full dosing]",
        ge=0.0,
        le=1.0,
    )

    # ========== Helper Methods ==========

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with feature names as keys.

        Includes timestamp (datetime) + 18 data features (float).
        """
        return self.model_dump()

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array in the order expected by the model.

        Returns:
            np.ndarray: Shape (18,) containing all features in order
        """
        return np.array([
            self.iGlob,
            self.tOut,
            self.vpOut,
            self.co2Out,
            self.wind,
            self.tSky,
            self.tSoOut,
            self.tAir,
            self.vpAir,
            self.co2Air,
            self.shScr,
            self.blScr,
            self.roof,
            self.tPipe,
            self.tGroPipe,
            self.lamp,
            self.intLamp,
            self.extCo2,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModelInput":
        """
        Create BaseModelInput from dictionary.

        Args:
            data: Dictionary with feature names as keys

        Returns:
            BaseModelInput instance
        """
        return cls(**data)

    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: datetime) -> "BaseModelInput":
        """
        Create BaseModelInput from numpy array.

        Args:
            arr: Numpy array of shape (18,) with features in order
            timestamp: Timestamp for this measurement

        Returns:
            BaseModelInput instance
        """
        if arr.shape != (18,):
            raise ValueError(f"Expected array of shape (18,), got {arr.shape}")

        return cls(
            timestamp=timestamp,
            iGlob=float(arr[0]),
            tOut=float(arr[1]),
            vpOut=float(arr[2]),
            co2Out=float(arr[3]),
            wind=float(arr[4]),
            tSky=float(arr[5]),
            tSoOut=float(arr[6]),
            tAir=float(arr[7]),
            vpAir=float(arr[8]),
            co2Air=float(arr[9]),
            shScr=float(arr[10]),
            blScr=float(arr[11]),
            roof=float(arr[12]),
            tPipe=float(arr[13]),
            tGroPipe=float(arr[14]),
            lamp=float(arr[15]),
            intLamp=float(arr[16]),
            extCo2=float(arr[17]),
        )

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get ordered list of feature names."""
        return [
            'iGlob', 'tOut', 'vpOut', 'co2Out', 'wind', 'tSky', 'tSoOut',
            'tAir', 'vpAir', 'co2Air', 'shScr', 'blScr', 'roof', 'tPipe',
            'tGroPipe', 'lamp', 'intLamp', 'extCo2'
        ]

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"BaseModelInput({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})\n"
            f"  Weather: iGlob={self.iGlob:.1f}W/m², tOut={self.tOut:.1f}°C, "
            f"vpOut={self.vpOut:.0f}Pa, co2Out={self.co2Out:.0f}ppm\n"
            f"  Indoor: tAir={self.tAir:.1f}°C, vpAir={self.vpAir:.0f}Pa, "
            f"co2Air={self.co2Air:.0f}ppm\n"
            f"  Controls: roof={self.roof:.2f}, shScr={self.shScr:.2f}, "
            f"tPipe={self.tPipe:.1f}°C, lamp={self.lamp:.2f}"
        )
