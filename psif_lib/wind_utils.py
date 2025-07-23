import numpy as np

def assign_wind_to_sectors(wind_direction):
    """
    Assign wind directions to 8 compass sectors.

    Parameters and Returns as in your existing function...
    """
    sector_bins = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    sectors = np.digitize(wind_direction, sector_bins, right=True) + 1
    sectors = np.where((wind_direction >= 337.5) | (wind_direction < 22.5), 1, sectors)
    return sectors.astype(np.int8)
