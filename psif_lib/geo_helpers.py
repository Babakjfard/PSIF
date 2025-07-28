# -----------------------------------------------------------------------------
#  Geodesy helpers
# -----------------------------------------------------------------------------

import numpy as np
from math import radians, sin, cos, atan2, sqrt

R_EARTH_KM = 6_371.0088  # mean Earth radius

def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised great‑circle distance in kilometres (lat/lon degrees)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1: np.ndarray, lon1: np.ndarray,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Initial bearing from (lat1,lon1) to (lat2,lon2) in degrees [0‑360)."""
    phi1, phi2 = map(np.radians, (lat1, lat2))
    dl = np.radians(lon2) - np.radians(lon1)
    y = np.sin(dl) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360