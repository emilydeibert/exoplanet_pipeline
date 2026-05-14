"""
Author: Emily Deibert
Last Modified: 12-10-2023
Description: Tools to extract metadata from GHOST files.
"""

from astropy import units as u 
from astropy.time import Time 
import numpy as np

def orbital_phase(T0, t, P0):
    """
    Orbital phase in [0, 1), where
    phase = 0   -> mid-transit
    phase = 0.5 -> secondary eclipse
    """
    phase = ((t - T0) / P0).to_value(u.dimensionless_unscaled)
    return np.mod(phase, 1.0)