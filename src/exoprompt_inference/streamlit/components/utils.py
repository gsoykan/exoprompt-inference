"""Utility functions for greenhouse climate calculations."""

import numpy as np
import torch
from torch import Tensor


def vp_to_rh(vp: Tensor, temp: Tensor) -> Tensor:
    """
    Convert vapor pressure and temperature to relative humidity.

    Args:
        vp: Vapor pressure in Pa
        temp: Temperature in °C

    Returns:
        Relative humidity in % (0-100)
    """

    def sat_vp(temp: Tensor) -> Tensor:
        """Calculate saturation vapor pressure using Magnus formula."""
        p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        sat = p[0] * torch.exp(p[2] * temp / (temp + p[1]))
        return sat

    rh = 100 * (vp / sat_vp(temp))

    # Handle numerical issues (bad models can produce inf/nan values)
    # float16 (Half precision) has max value ~65,504
    if torch.isinf(rh).any() or torch.isnan(rh).any():
        orig_dtype = vp.dtype
        dtype_bounds = {
            torch.float16: (-65504.0, 65504.0),
            torch.float32: (-3.4e38, 3.4e38),
            torch.float64: (-1.7e308, 1.7e308),
        }
        min_val, max_val = dtype_bounds.get(
            orig_dtype, (-float("inf"), float("inf"))
        )
        # Recalculate in float32 for better precision
        vp_32 = vp.to(torch.float32)
        temp_32 = temp.to(torch.float32)
        rh = 100 * (vp_32 / sat_vp(temp_32))
        # Clamp values to valid range
        rh = torch.clamp(rh, min=min_val, max=max_val)
        # Cast back to original dtype
        rh = rh.to(orig_dtype)

    return rh


def add_rh_to_unscaled(unscaled: np.ndarray) -> np.ndarray:
    """
    Calculate and append relative humidity to prediction array.

    Takes predictions/history with [tAir, vpAir, co2Air] and appends
    calculated rhAir as a 4th column.

    Args:
        unscaled: Array of shape [..., 3] with columns [tAir (°C), vpAir (Pa), co2Air (ppm)]

    Returns:
        Array of shape [..., 4] with columns [tAir, vpAir, co2Air, rhAir (%)]

    Example:
        >>> predictions = np.array([[20.0, 1500.0, 800.0], [22.0, 1600.0, 850.0]])
        >>> with_rh = add_rh_to_unscaled(predictions)
        >>> with_rh.shape
        (2, 4)
    """
    # Calculate RH from vpAir (index 1) and tAir (index 0)
    rh_tensor = vp_to_rh(
        vp=torch.from_numpy(unscaled[:, 1]),
        temp=torch.from_numpy(unscaled[:, 0])
    )

    # Append RH as 4th column
    unscaled_with_rh = np.concatenate(
        (unscaled, rh_tensor.unsqueeze(dim=1).cpu().numpy()),
        axis=1
    )

    return unscaled_with_rh
