"""
Exogenous prompt input for greenhouse climate model.

This module defines the standardized input format for greenhouse model parameters.
All parameter sources (JSON files, calibration results, etc.) should be converted
to this format before being used for inference.
"""

from typing import Dict, Any, List, Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class ExoPromptInput(BaseModel):
    """
    Standardized representation for greenhouse model exogenous parameters.

    Contains up to 254 parameters representing physical constants, greenhouse geometry,
    material properties, control strategies, and crop model parameters.

    Design Philosophy:
    - Raw parameter values (not scaled)
    - All fields are Optional to support parameter subsets (e.g., exo_params_to_take)
    - to_array() returns only non-None parameters sorted by key name (matching dataset behavior)
    - to_dict() returns only non-None fields
    - Scaling happens in preprocessing layer, not here

    Note: lambdaShScrPer is excluded from exoprompt (removed in dataset preprocessing)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # ========== Physical Constants ==========

    alfaLeafAir: Optional[float] = Field(
        default=None,
        description="Convective heat exchange coefficient leaf-air [W/m²/K]",
        ge=0.0,
        le=20.0,
    )
    L: Optional[float] = Field(
        default=None,
        description="Latent heat of evaporation [J/kg]",
        ge=2.0e6,
        le=3.0e6,
    )
    sigma: Optional[float] = Field(
        default=None,
        description="Stefan-Boltzmann constant [W/m²/K⁴]",
        ge=5.0e-8,
        le=6.0e-8,
    )
    epsCan: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of canopy [-]",
        ge=0.0,
        le=1.0,
    )
    epsSky: Optional[float] = Field(
        default=None, description="FIR emission coefficient of sky [-]", ge=0.0, le=1.0
    )
    etaGlobNir: Optional[float] = Field(
        default=None, description="Ratio of NIR in global radiation [-]", ge=0.0, le=1.0
    )
    etaGlobPar: Optional[float] = Field(
        default=None, description="Ratio of PAR in global radiation [-]", ge=0.0, le=1.0
    )
    etaMgPpm: Optional[float] = Field(
        default=None,
        description="CO2 conversion factor mg/m³ to ppm [-]",
        ge=0.0,
        le=1.0,
    )
    etaRoofThr: Optional[float] = Field(
        default=None,
        description="Roof ventilation discharge coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    rhoAir0: Optional[float] = Field(
        default=None, description="Density of air at sea level [kg/m³]", ge=0.5, le=2.0
    )
    rhoCanPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of canopy [-]",
        ge=0.0,
        le=1.0,
    )
    rhoCanNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of canopy [-]",
        ge=0.0,
        le=1.0,
    )
    rhoSteel: Optional[float] = Field(
        default=None, description="Density of steel [kg/m³]", ge=5000.0, le=10000.0
    )
    rhoWater: Optional[float] = Field(
        default=None, description="Density of water [kg/m³]", ge=900.0, le=1100.0
    )
    gamma: Optional[float] = Field(
        default=None, description="Psychrometric constant [Pa/K]", ge=50.0, le=100.0
    )
    omega: Optional[float] = Field(
        default=None,
        description="Angular velocity of earth rotation [rad/s]",
        ge=1.0e-7,
        le=3.0e-7,
    )

    # ========== Heat Capacity & Thermal Properties ==========

    capLeaf: Optional[float] = Field(
        default=None, description="Heat capacity of leaf [J/kg/K]", ge=500.0, le=2000.0
    )
    cEvap1: Optional[float] = Field(
        default=None, description="Evaporation coefficient 1 [-]", ge=0.0, le=10.0
    )
    cEvap2: Optional[float] = Field(
        default=None, description="Evaporation coefficient 2 [-]", ge=0.0, le=2.0
    )
    cEvap3Day: Optional[float] = Field(
        default=None,
        description="Evaporation coefficient 3 day [kg/m²/s/Pa]",
        ge=0.0,
        le=1.0e-5,
    )
    cEvap3Night: Optional[float] = Field(
        default=None,
        description="Evaporation coefficient 3 night [kg/m²/s/Pa]",
        ge=0.0,
        le=1.0e-10,
    )
    cEvap4Day: Optional[float] = Field(
        default=None,
        description="Evaporation coefficient 4 day [Pa⁻¹]",
        ge=0.0,
        le=1.0e-4,
    )
    cEvap4Night: Optional[float] = Field(
        default=None,
        description="Evaporation coefficient 4 night [Pa⁻¹]",
        ge=0.0,
        le=1.0e-4,
    )
    cPAir: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of air [J/kg/K]",
        ge=500.0,
        le=1500.0,
    )
    cPSteel: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of steel [J/kg/K]",
        ge=400.0,
        le=1000.0,
    )
    cPWater: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of water [J/kg/K]",
        ge=3000.0,
        le=5000.0,
    )

    # ========== Environmental & Geometry Constants ==========

    g: Optional[float] = Field(
        default=None, description="Gravitational acceleration [m/s²]", ge=9.0, le=10.0
    )
    hSo1: Optional[float] = Field(
        default=None, description="Thickness of soil layer 1 [m]", ge=0.0, le=1.0
    )
    hSo2: Optional[float] = Field(
        default=None, description="Thickness of soil layer 2 [m]", ge=0.0, le=1.0
    )
    hSo3: Optional[float] = Field(
        default=None, description="Thickness of soil layer 3 [m]", ge=0.0, le=1.0
    )
    hSo4: Optional[float] = Field(
        default=None, description="Thickness of soil layer 4 [m]", ge=0.0, le=1.0
    )
    hSo5: Optional[float] = Field(
        default=None, description="Thickness of soil layer 5 [m]", ge=0.0, le=2.0
    )

    # ========== Radiation Properties ==========

    k1Par: Optional[float] = Field(
        default=None, description="PAR extinction coefficient [-]", ge=0.0, le=2.0
    )
    k2Par: Optional[float] = Field(
        default=None, description="PAR extinction coefficient 2 [-]", ge=0.0, le=2.0
    )
    kNir: Optional[float] = Field(
        default=None, description="NIR extinction coefficient [-]", ge=0.0, le=2.0
    )
    kFir: Optional[float] = Field(
        default=None, description="FIR extinction coefficient [-]", ge=0.0, le=2.0
    )

    # ========== Molecular Properties ==========

    mAir: Optional[float] = Field(
        default=None, description="Molar mass of air [g/mol]", ge=20.0, le=40.0
    )
    hSoOut: Optional[float] = Field(
        default=None, description="Thickness of external soil layer [m]", ge=0.0, le=5.0
    )
    mWater: Optional[float] = Field(
        default=None, description="Molar mass of water [g/mol]", ge=10.0, le=25.0
    )
    R: Optional[float] = Field(
        default=None, description="Molar gas constant [J/kmol/K]", ge=8000.0, le=9000.0
    )

    # ========== Canopy Resistance ==========

    rCanSp: Optional[float] = Field(
        default=None, description="Canopy resistance setpoint [s/m]", ge=0.0, le=500.0
    )
    rB: Optional[float] = Field(
        default=None, description="Boundary layer resistance [s/m]", ge=0.0, le=500.0
    )
    rSMin: Optional[float] = Field(
        default=None, description="Minimum stomatal resistance [s/m]", ge=0.0, le=500.0
    )
    sRs: Optional[float] = Field(
        default=None, description="Slope of stomatal resistance [-]", ge=-10.0, le=10.0
    )
    etaGlobAir: Optional[float] = Field(
        default=None,
        description="Ratio of global radiation absorbed by air [-]",
        ge=0.0,
        le=1.0,
    )
    psi: Optional[float] = Field(
        default=None,
        description="Mean greenhouse cover slope [degrees]",
        ge=0.0,
        le=90.0,
    )

    # ========== Greenhouse Dimensions ==========

    aFlr: Optional[float] = Field(
        default=None, description="Floor area [m²]", ge=10.0, le=10000.0
    )
    aCov: Optional[float] = Field(
        default=None, description="Cover area [m²]", ge=10.0, le=20000.0
    )
    hAir: Optional[float] = Field(
        default=None, description="Height of main compartment [m]", ge=2.0, le=20.0
    )
    hGh: Optional[float] = Field(
        default=None, description="Mean greenhouse height [m]", ge=2.0, le=20.0
    )

    # ========== Heat Exchange Coefficients ==========

    cHecIn: Optional[float] = Field(
        default=None,
        description="Internal heat exchange coefficient [W/m²/K]",
        ge=0.0,
        le=10.0,
    )
    cHecOut1: Optional[float] = Field(
        default=None,
        description="External heat exchange coefficient 1 [W/m²/K]",
        ge=0.0,
        le=10.0,
    )
    cHecOut2: Optional[float] = Field(
        default=None,
        description="External heat exchange coefficient 2 [J/m³/K]",
        ge=0.0,
        le=5.0,
    )
    cHecOut3: Optional[float] = Field(
        default=None,
        description="External heat exchange coefficient 3 [-]",
        ge=0.0,
        le=5.0,
    )
    hElevation: Optional[float] = Field(
        default=None,
        description="Altitude of greenhouse [m above sea level]",
        ge=0.0,
        le=3000.0,
    )

    # ========== Ventilation Properties ==========

    aRoof: Optional[float] = Field(
        default=None,
        description="Maximum roof ventilation area [m²]",
        ge=0.0,
        le=1000.0,
    )
    hVent: Optional[float] = Field(
        default=None,
        description="Vertical dimension of single ventilation opening [m]",
        ge=0.0,
        le=5.0,
    )
    etaInsScr: Optional[float] = Field(
        default=None, description="Insulation screen porosity [-]", ge=0.0, le=1.0
    )
    aSide: Optional[float] = Field(
        default=None, description="Side ventilation area [m²]", ge=0.0, le=1000.0
    )
    cDgh: Optional[float] = Field(
        default=None,
        description="Ventilation discharge coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    cLeakage: Optional[float] = Field(
        default=None, description="Leakage coefficient [-]", ge=0.0, le=1.0e-3
    )
    cWgh: Optional[float] = Field(
        default=None,
        description="Ventilation global wind coefficient [-]",
        ge=0.0,
        le=0.1,
    )
    hSideRoof: Optional[float] = Field(
        default=None,
        description="Vertical distance between side and roof vents [m]",
        ge=0.0,
        le=10.0,
    )

    # ========== Roof Properties ==========

    epsRfFir: Optional[float] = Field(
        default=None, description="FIR emission coefficient of roof [-]", ge=0.0, le=1.0
    )
    rhoRf: Optional[float] = Field(
        default=None, description="Density of roof [kg/m³]", ge=1000.0, le=5000.0
    )
    rhoRfNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of roof [-]",
        ge=0.0,
        le=1.0,
    )
    rhoRfPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of roof [-]",
        ge=0.0,
        le=1.0,
    )
    rhoRfFir: Optional[float] = Field(
        default=None,
        description="FIR reflection coefficient of roof [-]",
        ge=0.0,
        le=1.0,
    )
    tauRfNir: Optional[float] = Field(
        default=None,
        description="NIR transmission coefficient of roof [-]",
        ge=0.0,
        le=1.0,
    )
    tauRfPar: Optional[float] = Field(
        default=None,
        description="PAR transmission coefficient of roof [-]",
        ge=0.0,
        le=1.0,
    )
    tauRfFir: Optional[float] = Field(
        default=None,
        description="FIR transmission coefficient of roof [-]",
        ge=0.0,
        le=1.0,
    )
    lambdaRf: Optional[float] = Field(
        default=None, description="Thermal conductivity of roof [W/m/K]", ge=0.0, le=5.0
    )
    cPRf: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of roof [J/kg/K]",
        ge=500.0,
        le=2000.0,
    )
    hRf: Optional[float] = Field(
        default=None, description="Thickness of roof [m]", ge=0.0, le=0.1
    )

    # ========== Permanent Shade Screen Properties ==========

    epsShScrPerFir: Optional[float] = Field(
        default=None,
        description="FIR emission of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoShScrPer: Optional[float] = Field(
        default=None,
        description="Density of permanent shade screen [kg/m³]",
        ge=0.0,
        le=1000.0,
    )
    rhoShScrPerNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoShScrPerPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoShScrPerFir: Optional[float] = Field(
        default=None,
        description="FIR reflection coefficient of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauShScrPerNir: Optional[float] = Field(
        default=None,
        description="NIR transmission of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauShScrPerPar: Optional[float] = Field(
        default=None,
        description="PAR transmission of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauShScrPerFir: Optional[float] = Field(
        default=None,
        description="FIR transmission of permanent shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    cPShScrPer: Optional[float] = Field(
        default=None,
        description="Heat capacity of permanent shade screen [J/kg/K]",
        ge=0.0,
        le=2000.0,
    )
    hShScrPer: Optional[float] = Field(
        default=None,
        description="Thickness of permanent shade screen [m]",
        ge=0.0,
        le=0.01,
    )

    # ========== Movable Shade Screen Properties ==========

    rhoShScrNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of movable shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoShScrPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of movable shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoShScrFir: Optional[float] = Field(
        default=None,
        description="FIR reflection coefficient of movable shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauShScrNir: Optional[float] = Field(
        default=None,
        description="NIR transmission of movable shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauShScrPar: Optional[float] = Field(
        default=None,
        description="PAR transmission of movable shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauShScrFir: Optional[float] = Field(
        default=None,
        description="FIR transmission of movable shade screen [-]",
        ge=0.0,
        le=1.0,
    )
    etaShScrCd: Optional[float] = Field(
        default=None,
        description="Effect of movable shade screen on ventilation discharge [-]",
        ge=0.0,
        le=1.0,
    )
    etaShScrCw: Optional[float] = Field(
        default=None,
        description="Effect of movable shade screen on wind [-]",
        ge=0.0,
        le=1.0,
    )
    kShScr: Optional[float] = Field(
        default=None,
        description="Radiation extinction coefficient of movable shade screen [-]",
        ge=0.0,
        le=10.0,
    )

    # ========== Thermal Screen Properties ==========

    epsThScrFir: Optional[float] = Field(
        default=None, description="FIR emission of thermal screen [-]", ge=0.0, le=1.0
    )
    rhoThScr: Optional[float] = Field(
        default=None, description="Density of thermal screen [kg/m³]", ge=0.0, le=1000.0
    )
    rhoThScrNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of thermal screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoThScrPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of thermal screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoThScrFir: Optional[float] = Field(
        default=None,
        description="FIR reflection coefficient of thermal screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauThScrNir: Optional[float] = Field(
        default=None,
        description="NIR transmission of thermal screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauThScrPar: Optional[float] = Field(
        default=None,
        description="PAR transmission of thermal screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauThScrFir: Optional[float] = Field(
        default=None,
        description="FIR transmission of thermal screen [-]",
        ge=0.0,
        le=1.0,
    )
    cPThScr: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of thermal screen [J/kg/K]",
        ge=500.0,
        le=3000.0,
    )
    hThScr: Optional[float] = Field(
        default=None, description="Thickness of thermal screen [m]", ge=0.0, le=0.01
    )
    kThScr: Optional[float] = Field(
        default=None,
        description="Thermal screen flux coefficient [m³/m²/K/s]",
        ge=0.0,
        le=0.01,
    )

    # ========== Blackout Screen Properties ==========

    epsBlScrFir: Optional[float] = Field(
        default=None, description="FIR emission of blackout screen [-]", ge=0.0, le=1.0
    )
    rhoBlScr: Optional[float] = Field(
        default=None,
        description="Density of blackout screen [kg/m³]",
        ge=0.0,
        le=1000.0,
    )
    rhoBlScrNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of blackout screen [-]",
        ge=0.0,
        le=1.0,
    )
    rhoBlScrPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of blackout screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauBlScrNir: Optional[float] = Field(
        default=None,
        description="NIR transmission of blackout screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauBlScrPar: Optional[float] = Field(
        default=None,
        description="PAR transmission of blackout screen [-]",
        ge=0.0,
        le=1.0,
    )
    tauBlScrFir: Optional[float] = Field(
        default=None,
        description="FIR transmission of blackout screen [-]",
        ge=0.0,
        le=1.0,
    )
    cPBlScr: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of blackout screen [J/kg/K]",
        ge=500.0,
        le=3000.0,
    )
    hBlScr: Optional[float] = Field(
        default=None, description="Thickness of blackout screen [m]", ge=0.0, le=0.01
    )
    kBlScr: Optional[float] = Field(
        default=None,
        description="Blackout screen flux coefficient [m³/m²/K/s]",
        ge=0.0,
        le=0.01,
    )

    # ========== Floor Properties ==========

    epsFlr: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of floor [-]",
        ge=0.0,
        le=1.0,
    )
    rhoFlr: Optional[float] = Field(
        default=None, description="Density of floor [kg/m³]", ge=1000.0, le=5000.0
    )
    rhoFlrNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of floor [-]",
        ge=0.0,
        le=1.0,
    )
    rhoFlrPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of floor [-]",
        ge=0.0,
        le=1.0,
    )
    lambdaFlr: Optional[float] = Field(
        default=None,
        description="Thermal conductivity of floor [W/m/K]",
        ge=0.0,
        le=5.0,
    )
    cPFlr: Optional[float] = Field(
        default=None,
        description="Specific heat capacity of floor [J/kg/K]",
        ge=500.0,
        le=2000.0,
    )
    hFlr: Optional[float] = Field(
        default=None, description="Thickness of floor [m]", ge=0.0, le=0.5
    )

    # ========== Soil Properties ==========

    rhoCpSo: Optional[float] = Field(
        default=None,
        description="Volumetric heat capacity of soil [J/m³/K]",
        ge=1.0e6,
        le=3.0e6,
    )
    lambdaSo: Optional[float] = Field(
        default=None, description="Thermal conductivity of soil [W/m/K]", ge=0.0, le=5.0
    )

    # ========== Heating Pipe Properties ==========

    epsPipe: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of heating pipe [-]",
        ge=0.0,
        le=1.0,
    )
    phiPipeE: Optional[float] = Field(
        default=None,
        description="External diameter of heating pipe [m]",
        ge=0.0,
        le=0.2,
    )
    phiPipeI: Optional[float] = Field(
        default=None,
        description="Internal diameter of heating pipe [m]",
        ge=0.0,
        le=0.2,
    )
    lPipe: Optional[float] = Field(
        default=None,
        description="Length of heating pipe per floor area [m/m²]",
        ge=0.0,
        le=5.0,
    )
    pBoil: Optional[float] = Field(
        default=None, description="Capacity of boiler [W]", ge=0.0, le=1.0e7
    )

    # ========== CO2 Injection ==========

    phiExtCo2: Optional[float] = Field(
        default=None,
        description="Capacity of external CO2 source [mg/s]",
        ge=0.0,
        le=5000.0,
    )

    # ========== Computed/Derived Capacities ==========

    capPipe: Optional[float] = Field(
        default=None,
        description="Heat capacity of heating pipe [J/K]",
        ge=0.0,
        le=1.0e5,
    )
    rhoAir: Optional[float] = Field(
        default=None, description="Density of air [kg/m³]", ge=0.5, le=2.0
    )
    capAir: Optional[float] = Field(
        default=None, description="Heat capacity of air [J/K]", ge=0.0, le=1.0e6
    )
    capFlr: Optional[float] = Field(
        default=None, description="Heat capacity of floor [J/K]", ge=0.0, le=1.0e6
    )
    capSo1: Optional[float] = Field(
        default=None,
        description="Heat capacity of soil layer 1 [J/K]",
        ge=0.0,
        le=1.0e6,
    )
    capSo2: Optional[float] = Field(
        default=None,
        description="Heat capacity of soil layer 2 [J/K]",
        ge=0.0,
        le=1.0e6,
    )
    capSo3: Optional[float] = Field(
        default=None,
        description="Heat capacity of soil layer 3 [J/K]",
        ge=0.0,
        le=1.0e6,
    )
    capSo4: Optional[float] = Field(
        default=None,
        description="Heat capacity of soil layer 4 [J/K]",
        ge=0.0,
        le=1.0e6,
    )
    capSo5: Optional[float] = Field(
        default=None,
        description="Heat capacity of soil layer 5 [J/K]",
        ge=0.0,
        le=2.0e6,
    )
    capThScr: Optional[float] = Field(
        default=None,
        description="Heat capacity of thermal screen [J/K]",
        ge=0.0,
        le=1000.0,
    )
    capTop: Optional[float] = Field(
        default=None,
        description="Heat capacity of top compartment [J/K]",
        ge=0.0,
        le=1.0e5,
    )
    capBlScr: Optional[float] = Field(
        default=None,
        description="Heat capacity of blackout screen [J/K]",
        ge=0.0,
        le=1000.0,
    )
    capCo2Air: Optional[float] = Field(
        default=None, description="Capacity for CO2 in main air [m³]", ge=0.0, le=100.0
    )
    capCo2Top: Optional[float] = Field(
        default=None, description="Capacity for CO2 in top air [m³]", ge=0.0, le=50.0
    )
    aPipe: Optional[float] = Field(
        default=None,
        description="Surface area of heating pipe per floor area [m²/m²]",
        ge=0.0,
        le=1.0,
    )
    fCanFlr: Optional[float] = Field(
        default=None, description="View factor from canopy to floor [-]", ge=0.0, le=1.0
    )
    pressure: Optional[float] = Field(
        default=None, description="Atmospheric pressure [Pa]", ge=80000.0, le=110000.0
    )

    # ========== Photosynthesis Parameters ==========

    globJtoUmol: Optional[float] = Field(
        default=None,
        description="Conversion from global radiation to PAR [μmol/J]",
        ge=0.0,
        le=10.0,
    )
    j25LeafMax: Optional[float] = Field(
        default=None,
        description="Maximum rate of electron transport at 25°C [μmol/m²/s]",
        ge=0.0,
        le=500.0,
    )
    cGamma: Optional[float] = Field(
        default=None, description="CO2 compensation point [μmol/mol]", ge=0.0, le=10.0
    )
    etaCo2AirStom: Optional[float] = Field(
        default=None,
        description="Ratio of CO2 in stomata to CO2 in air [-]",
        ge=0.0,
        le=1.0,
    )
    eJ: Optional[float] = Field(
        default=None,
        description="Activation energy for Jmax [J/mol]",
        ge=20000.0,
        le=60000.0,
    )
    t25k: Optional[float] = Field(
        default=None,
        description="Reference temperature for photosynthesis [K]",
        ge=273.0,
        le=323.0,
    )
    S: Optional[float] = Field(
        default=None,
        description="Entropy term for electron transport [J/mol/K]",
        ge=500.0,
        le=1000.0,
    )
    H: Optional[float] = Field(
        default=None,
        description="Deactivation energy [J/mol]",
        ge=100000.0,
        le=300000.0,
    )
    theta: Optional[float] = Field(
        default=None,
        description="Degree of curvature of electron transport [-]",
        ge=0.0,
        le=1.0,
    )
    alpha: Optional[float] = Field(
        default=None,
        description="Conversion factor from photons to electrons [-]",
        ge=0.0,
        le=1.0,
    )

    # ========== Carbon Balance ==========

    mCh2o: Optional[float] = Field(
        default=None, description="Molar mass of CH2O [kg/mol]", ge=0.01, le=0.05
    )
    mCo2: Optional[float] = Field(
        default=None, description="Molar mass of CO2 [kg/mol]", ge=0.03, le=0.06
    )
    parJtoUmolSun: Optional[float] = Field(
        default=None,
        description="Conversion from sun PAR to μmol [μmol/J]",
        ge=0.0,
        le=10.0,
    )

    # ========== Crop Model Parameters ==========

    laiMax: Optional[float] = Field(
        default=None, description="Maximum leaf area index [m²/m²]", ge=0.0, le=10.0
    )
    sla: Optional[float] = Field(
        default=None, description="Specific leaf area [m²/mg]", ge=0.0, le=1.0e-3
    )
    rgr: Optional[float] = Field(
        default=None,
        description="Relative growth rate of leaf area [s⁻¹]",
        ge=0.0,
        le=1.0e-4,
    )
    cLeafMax: Optional[float] = Field(
        default=None, description="Maximum leaf capacity [mg/m²]", ge=0.0, le=1.0e6
    )
    cFruitMax: Optional[float] = Field(
        default=None, description="Maximum fruit capacity [mg/m²]", ge=0.0, le=1.0e6
    )
    cFruitG: Optional[float] = Field(
        default=None,
        description="Fruit growth respiration coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    cLeafG: Optional[float] = Field(
        default=None,
        description="Leaf growth respiration coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    cStemG: Optional[float] = Field(
        default=None,
        description="Stem growth respiration coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    cRgr: Optional[float] = Field(
        default=None,
        description="Regression coefficient for maintenance respiration [s⁻¹]",
        ge=0.0,
        le=1.0e7,
    )
    q10m: Optional[float] = Field(
        default=None,
        description="Q10 value for maintenance respiration [-]",
        ge=1.0,
        le=5.0,
    )
    cFruitM: Optional[float] = Field(
        default=None,
        description="Fruit maintenance respiration coefficient [s⁻¹]",
        ge=0.0,
        le=1.0e-5,
    )
    cLeafM: Optional[float] = Field(
        default=None,
        description="Leaf maintenance respiration coefficient [s⁻¹]",
        ge=0.0,
        le=1.0e-5,
    )
    cStemM: Optional[float] = Field(
        default=None,
        description="Stem maintenance respiration coefficient [s⁻¹]",
        ge=0.0,
        le=1.0e-5,
    )
    rgFruit: Optional[float] = Field(
        default=None,
        description="Potential fruit growth coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    rgLeaf: Optional[float] = Field(
        default=None,
        description="Potential leaf growth coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    rgStem: Optional[float] = Field(
        default=None,
        description="Potential stem growth coefficient [-]",
        ge=0.0,
        le=1.0,
    )
    cBufMax: Optional[float] = Field(
        default=None, description="Maximum buffer capacity [mg/m²]", ge=0.0, le=1.0e5
    )
    cBufMin: Optional[float] = Field(
        default=None, description="Minimum buffer capacity [mg/m²]", ge=0.0, le=1.0e5
    )

    # ========== Control Setpoints ==========

    tCan24Max: Optional[float] = Field(
        default=None,
        description="Maximum 24h mean canopy temperature [°C]",
        ge=0.0,
        le=50.0,
    )
    tCan24Min: Optional[float] = Field(
        default=None,
        description="Minimum 24h mean canopy temperature [°C]",
        ge=0.0,
        le=50.0,
    )
    tCanMax: Optional[float] = Field(
        default=None,
        description="Maximum instantaneous canopy temperature [°C]",
        ge=0.0,
        le=50.0,
    )
    tCanMin: Optional[float] = Field(
        default=None,
        description="Minimum instantaneous canopy temperature [°C]",
        ge=0.0,
        le=50.0,
    )
    tEndSum: Optional[float] = Field(
        default=None,
        description="Temperature sum for crop development [°C·day]",
        ge=0.0,
        le=5000.0,
    )
    rhMax: Optional[float] = Field(
        default=None, description="Maximum relative humidity [%]", ge=0.0, le=100.0
    )
    dayThresh: Optional[float] = Field(
        default=None,
        description="Threshold for day/night regime [W/m²]",
        ge=0.0,
        le=100.0,
    )
    tSpDay: Optional[float] = Field(
        default=None, description="Temperature setpoint day [°C]", ge=0.0, le=40.0
    )
    tSpNight: Optional[float] = Field(
        default=None, description="Temperature setpoint night [°C]", ge=0.0, le=40.0
    )
    tHeatBand: Optional[float] = Field(
        default=None,
        description="Temperature dead band for heating [K]",
        ge=-10.0,
        le=10.0,
    )
    tVentOff: Optional[float] = Field(
        default=None,
        description="Temperature offset for ventilation [K]",
        ge=-10.0,
        le=10.0,
    )
    tScreenOn: Optional[float] = Field(
        default=None,
        description="Temperature for screen closure [K]",
        ge=-10.0,
        le=10.0,
    )
    thScrSpDay: Optional[float] = Field(
        default=None, description="Thermal screen setpoint day [K]", ge=-10.0, le=20.0
    )
    thScrSpNight: Optional[float] = Field(
        default=None, description="Thermal screen setpoint night [K]", ge=-10.0, le=20.0
    )
    thScrPband: Optional[float] = Field(
        default=None,
        description="Proportional band for thermal screen [K]",
        ge=-10.0,
        le=10.0,
    )
    co2SpDay: Optional[float] = Field(
        default=None, description="CO2 setpoint day [ppm]", ge=300.0, le=2000.0
    )
    co2Band: Optional[float] = Field(
        default=None, description="CO2 proportional band [ppm]", ge=-500.0, le=500.0
    )
    heatDeadZone: Optional[float] = Field(
        default=None, description="Heating dead zone [K]", ge=0.0, le=20.0
    )
    ventHeatPband: Optional[float] = Field(
        default=None,
        description="Ventilation heating proportional band [K]",
        ge=0.0,
        le=20.0,
    )
    ventColdPband: Optional[float] = Field(
        default=None,
        description="Ventilation cooling proportional band [K]",
        ge=-20.0,
        le=20.0,
    )
    ventRhPband: Optional[float] = Field(
        default=None,
        description="Ventilation RH proportional band [%]",
        ge=0.0,
        le=50.0,
    )
    thScrRh: Optional[float] = Field(
        default=None, description="Thermal screen RH threshold [%]", ge=-100.0, le=100.0
    )
    thScrRhPband: Optional[float] = Field(
        default=None,
        description="Thermal screen RH proportional band [%]",
        ge=0.0,
        le=50.0,
    )
    thScrDeadZone: Optional[float] = Field(
        default=None, description="Thermal screen dead zone [K]", ge=0.0, le=20.0
    )

    # ========== Lamp Control ==========

    lampsOn: Optional[float] = Field(
        default=None, description="Time when lamps turn on [hour]", ge=0.0, le=24.0
    )
    lampsOff: Optional[float] = Field(
        default=None, description="Time when lamps turn off [hour]", ge=0.0, le=24.0
    )
    dayLampStart: Optional[float] = Field(
        default=None,
        description="PAR threshold for lamp activation [W/m²]",
        ge=-100.0,
        le=1000.0,
    )
    dayLampStop: Optional[float] = Field(
        default=None,
        description="PAR threshold for lamp deactivation [W/m²]",
        ge=0.0,
        le=1000.0,
    )
    lampsOffSun: Optional[float] = Field(
        default=None,
        description="Global radiation threshold to turn lamps off [W/m²]",
        ge=0.0,
        le=1000.0,
    )
    lampRadSumLimit: Optional[float] = Field(
        default=None,
        description="Daily radiation sum limit for lamps [MJ/m²/day]",
        ge=0.0,
        le=100.0,
    )
    lampExtraHeat: Optional[float] = Field(
        default=None,
        description="Extra heating requirement when using lamps [K]",
        ge=0.0,
        le=20.0,
    )
    blScrExtraRh: Optional[float] = Field(
        default=None,
        description="Extra RH threshold for blackout screen [%]",
        ge=0.0,
        le=100.0,
    )
    useBlScr: Optional[float] = Field(
        default=None, description="Flag to use blackout screen [0 or 1]", ge=0.0, le=1.0
    )

    # ========== Mechanical Climate Control ==========

    mechCoolPband: Optional[float] = Field(
        default=None,
        description="Mechanical cooling proportional band [K]",
        ge=0.0,
        le=20.0,
    )
    mechDehumidPband: Optional[float] = Field(
        default=None,
        description="Mechanical dehumidification proportional band [%]",
        ge=0.0,
        le=50.0,
    )
    heatBufPband: Optional[float] = Field(
        default=None, description="Heat buffer proportional band [K]", ge=-20.0, le=20.0
    )
    mechCoolDeadZone: Optional[float] = Field(
        default=None, description="Mechanical cooling dead zone [K]", ge=0.0, le=20.0
    )

    # ========== Grow Pipe Properties ==========

    epsGroPipe: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of grow pipe [-]",
        ge=0.0,
        le=1.0,
    )
    lGroPipe: Optional[float] = Field(
        default=None,
        description="Length of grow pipe per floor area [m/m²]",
        ge=0.0,
        le=5.0,
    )
    phiGroPipeE: Optional[float] = Field(
        default=None, description="External diameter of grow pipe [m]", ge=0.0, le=0.2
    )
    phiGroPipeI: Optional[float] = Field(
        default=None, description="Internal diameter of grow pipe [m]", ge=0.0, le=0.2
    )
    aGroPipe: Optional[float] = Field(
        default=None,
        description="Surface area of grow pipe per floor area [m²/m²]",
        ge=0.0,
        le=1.0,
    )
    pBoilGro: Optional[float] = Field(
        default=None, description="Capacity of grow pipe boiler [W]", ge=0.0, le=1.0e7
    )
    capGroPipe: Optional[float] = Field(
        default=None, description="Heat capacity of grow pipe [J/K]", ge=0.0, le=1.0e5
    )

    # ========== Lamp Properties ==========

    thetaLampMax: Optional[float] = Field(
        default=None,
        description="Maximum intensity of lamps [μmol/m²/s]",
        ge=0.0,
        le=500.0,
    )
    heatCorrection: Optional[float] = Field(
        default=None,
        description="Correction factor for lamp heat [K]",
        ge=-10.0,
        le=10.0,
    )
    etaLampPar: Optional[float] = Field(
        default=None, description="Fraction of lamp energy as PAR [-]", ge=0.0, le=1.0
    )
    etaLampNir: Optional[float] = Field(
        default=None, description="Fraction of lamp energy as NIR [-]", ge=0.0, le=1.0
    )
    tauLampPar: Optional[float] = Field(
        default=None,
        description="PAR transmission coefficient of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    rhoLampPar: Optional[float] = Field(
        default=None,
        description="PAR reflection coefficient of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    tauLampNir: Optional[float] = Field(
        default=None,
        description="NIR transmission coefficient of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    rhoLampNir: Optional[float] = Field(
        default=None,
        description="NIR reflection coefficient of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    tauLampFir: Optional[float] = Field(
        default=None,
        description="FIR transmission coefficient of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    aLamp: Optional[float] = Field(
        default=None,
        description="Surface area of lamps per floor area [m²/m²]",
        ge=0.0,
        le=0.5,
    )
    epsLampTop: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of top side of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    epsLampBottom: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of bottom side of lamp [-]",
        ge=0.0,
        le=1.0,
    )
    capLamp: Optional[float] = Field(
        default=None, description="Heat capacity of lamp [J/K/m²]", ge=0.0, le=1000.0
    )
    cHecLampAir: Optional[float] = Field(
        default=None,
        description="Heat exchange coefficient lamp-air [W/m²/K]",
        ge=0.0,
        le=10.0,
    )
    etaLampCool: Optional[float] = Field(
        default=None,
        description="Fraction of lamp energy removed by cooling [-]",
        ge=0.0,
        le=1.0,
    )
    zetaLampPar: Optional[float] = Field(
        default=None,
        description="J to μmol conversion for lamp PAR [μmol/J]",
        ge=0.0,
        le=10.0,
    )

    # ========== Interlighting Lamp Properties ==========

    vIntLampPos: Optional[float] = Field(
        default=None,
        description="Vertical position of interlighting [-]",
        ge=0.0,
        le=1.0,
    )
    fIntLampDown: Optional[float] = Field(
        default=None,
        description="Fraction of interlighting PAR going down [-]",
        ge=0.0,
        le=1.0,
    )
    capIntLamp: Optional[float] = Field(
        default=None,
        description="Heat capacity of interlighting lamp [J/K/m²]",
        ge=0.0,
        le=1000.0,
    )
    etaIntLampPar: Optional[float] = Field(
        default=None,
        description="Fraction of interlighting energy as PAR [-]",
        ge=0.0,
        le=1.0,
    )
    etaIntLampNir: Optional[float] = Field(
        default=None,
        description="Fraction of interlighting energy as NIR [-]",
        ge=0.0,
        le=1.0,
    )
    aIntLamp: Optional[float] = Field(
        default=None,
        description="Surface area of interlighting per floor area [m²/m²]",
        ge=0.0,
        le=0.5,
    )
    epsIntLamp: Optional[float] = Field(
        default=None,
        description="FIR emission coefficient of interlighting [-]",
        ge=0.0,
        le=1.0,
    )
    thetaIntLampMax: Optional[float] = Field(
        default=None,
        description="Maximum intensity of interlighting [μmol/m²/s]",
        ge=0.0,
        le=500.0,
    )
    zetaIntLampPar: Optional[float] = Field(
        default=None,
        description="J to μmol conversion for interlighting PAR [μmol/J]",
        ge=0.0,
        le=10.0,
    )
    cHecIntLampAir: Optional[float] = Field(
        default=None,
        description="Heat exchange coefficient interlighting-air [W/m²/K]",
        ge=0.0,
        le=10.0,
    )
    tauIntLampFir: Optional[float] = Field(
        default=None,
        description="FIR transmission coefficient of interlighting [-]",
        ge=0.0,
        le=1.0,
    )

    # ========== Canopy Extinction Coefficients (for interlighting) ==========

    k1IntPar: Optional[float] = Field(
        default=None,
        description="PAR extinction coefficient 1 for interlighting [-]",
        ge=0.0,
        le=5.0,
    )
    k2IntPar: Optional[float] = Field(
        default=None,
        description="PAR extinction coefficient 2 for interlighting [-]",
        ge=0.0,
        le=5.0,
    )
    kIntNir: Optional[float] = Field(
        default=None,
        description="NIR extinction coefficient for interlighting [-]",
        ge=0.0,
        le=5.0,
    )
    kIntFir: Optional[float] = Field(
        default=None,
        description="FIR extinction coefficient for interlighting [-]",
        ge=0.0,
        le=5.0,
    )

    # ========== Additional Ventilation Parameters ==========

    cLeakTop: Optional[float] = Field(
        default=None, description="Leakage to top compartment [-]", ge=0.0, le=1.0
    )
    minWind: Optional[float] = Field(
        default=None, description="Minimum wind speed [m/s]", ge=0.0, le=5.0
    )

    # ========== Helper Methods ==========

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary with parameter names as keys.

        Returns only non-None parameters (matching dataset behavior).
        Useful for partial parameter sets (e.g., exo_params_to_take).
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}

    @staticmethod
    def dict_to_array(data_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert parameter dictionary to sorted numpy array.

        Static helper for converting any parameter dict (e.g., scaled values)
        to array format. Useful when you have a transformed/scaled dict that
        you want to convert to array without creating an ExoPromptInput instance.

        Args:
            data_dict: Dictionary with parameter names as keys and values

        Returns:
            np.ndarray: Shape (N,) where N is number of parameters in dict.
                       Keys are sorted alphabetically before conversion.

        Example:
            >>> scaled_params = {"alfaLeafAir": 0.5, "L": 0.8}
            >>> arr = ExoPromptInput.dict_to_array(scaled_params)
            >>> arr.shape
            (2,)
        """
        sorted_keys = sorted(data_dict.keys())
        return np.array([data_dict[k] for k in sorted_keys], dtype=np.float32)

    def to_array(self) -> np.ndarray:
        """
        Convert this instance's parameters to numpy array in sorted key order.

        Returns:
            np.ndarray: Shape (N,) where N is number of non-None parameters.
                       For full parameter set, N=254. For partial sets (e.g., exo_params_to_take), N<254.

        Example:
            >>> exo = ExoPromptInput(alfaLeafAir=5.0, L=2.45e6)
            >>> arr = exo.to_array()
            >>> arr.shape
            (2,)
        """
        return ExoPromptInput.dict_to_array(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExoPromptInput":
        """
        Create ExoPromptInput from dictionary.

        Args:
            data: Dictionary with parameter names as keys

        Returns:
            ExoPromptInput instance
        """
        return cls(**data)

    @classmethod
    def from_array(cls, arr: np.ndarray, param_names: List[str]) -> "ExoPromptInput":
        """
        Create ExoPromptInput from numpy array.

        Args:
            arr: Numpy array of shape (N,) with parameters
            param_names: List of parameter names in same order as array (sorted)

        Returns:
            ExoPromptInput instance
        """
        if len(arr) != len(param_names):
            raise ValueError(
                f"Array length {len(arr)} doesn't match param_names length {len(param_names)}"
            )

        data = {name: float(val) for name, val in zip(param_names, arr)}
        return cls(**data)

    def get_present_param_names(self) -> List[str]:
        """
        Get list of non-None parameter names for this instance in sorted order.

        Returns:
            List of present parameter names (sorted alphabetically).
            Length varies based on which parameters are set.
        """
        data_dict = self.to_dict()
        return sorted(data_dict.keys())

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Get list of all possible parameter names in sorted order.

        Returns:
            List of all 254 possible parameter names (sorted alphabetically),
            regardless of which are actually present in any specific instance.
        """
        # Get all field names from the model
        field_names = list(cls.model_fields.keys())
        # Sort alphabetically (matching dataset behavior)
        return sorted(field_names)

    def __str__(self) -> str:
        """Human-readable string representation."""

        # Helper to format optional floats
        def fmt(val, spec):
            return format(val, spec) if val is not None else "None"

        num_present = len(self.to_dict())
        return (
            f"ExoPromptInput ({num_present}/254 parameters present):\n"
            f"  Physical: alfaLeafAir={fmt(self.alfaLeafAir, '.1f')}, "
            f"L={fmt(self.L, '.1e')}, sigma={fmt(self.sigma, '.2e')}\n"
            f"  Geometry: aFlr={fmt(self.aFlr, '.1f')}m², hAir={fmt(self.hAir, '.1f')}m, "
            f"hGh={fmt(self.hGh, '.1f')}m\n"
            f"  Control: tSpDay={fmt(self.tSpDay, '.1f')}°C, tSpNight={fmt(self.tSpNight, '.1f')}°C, "
            f"co2SpDay={fmt(self.co2SpDay, '.0f')}ppm\n"
            f"  Heating: pBoil={fmt(self.pBoil, '.0f')}W, pBoilGro={fmt(self.pBoilGro, '.0f')}W\n"
            f"  Lamps: thetaLampMax={fmt(self.thetaLampMax, '.0f')}μmol/m²/s, "
            f"etaLampPar={fmt(self.etaLampPar, '.2f')}\n"
            f"  Crop: laiMax={fmt(self.laiMax, '.1f')}, cLeafMax={fmt(self.cLeafMax, '.0f')}mg/m²"
        )
