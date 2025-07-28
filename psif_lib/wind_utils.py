import numpy as np
import xarray as xr

def assign_wind_to_sectors(wind_direction):
    """
    Assign wind directions to 8 compass sectors.

    Parameters and Returns as in your existing function...
    """
    sector_bins = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    sectors = np.digitize(wind_direction, sector_bins, right=True) + 1
    sectors = np.where((wind_direction >= 337.5) | (wind_direction < 22.5), 1, sectors)
    return sectors.astype(np.int8)

def get_wind_at_fire(ds_wind, fires, suffix=''):
    """
    Assign wind statistics to each fire event by locating the nearest grid cell in the wind dataset.

    For each fire in the `fires` dataset (or DataFrame), this function looks up the wind variables
    from the xarray Dataset `ds_wind` at the fire's acquisition date and location (latitude and longitude).
    It uses nearest-neighbor selection to find the closest spatial and temporal point in the wind data.

    The function then adds columns representing wind sector durations (e.g., duration_sector_1, ..., duration_sector_8)
    from `ds_wind` to the `fires` data, optionally appending a suffix to the column names.

    Parameters
    ----------
    ds_wind : xarray.Dataset
        An xarray Dataset containing wind-related variables indexed by time, latitude, and longitude.
        Must include variables named like "duration_sector_1" through "duration_sector_8".
    
    fires : pandas.DataFrame or xarray.Dataset
        A dataset containing fire events with columns or variables:
        - 'latitude' (float): latitude of the fire location
        - 'longitude' (float): longitude of the fire location
        - 'acq_date' (datetime-like): acquisition date/time of the fire event
    
    suffix : str, optional
        A string suffix to append to the new wind sector columns added to `fires`.
        This can be useful to distinguish multiple wind datasets or scenarios. 
        NOTE: distinction of wind at origin (fire) vs. destination (BlockGroup) is necessary. This is 
        to make this distinction by adding suffix (e.g. "_BG" for BlockGroups)
        Default is '' (no suffix).

    Returns
    -------
    fires : same type as input
        The input `fires` dataset augmented with new columns named `"duration_sector_1"`, ..., `"duration_sector_8"`
        (plus optional suffix), containing the wind sector duration values matched by nearest space/time point.

    Notes
    -----
    - The selection uses xarray's `.sel(..., method='nearest')` functionality.
    - The function assumes the coordinates in `ds_wind` and the fire locations use compatible units and reference systems.
    """
    # Build a 2‑D indexer
    latitudes  = xr.DataArray(fires['latitude'].values, dims="fire")
    longitudes = xr.DataArray(fires['longitude'].values, dims="fire")
    times      = xr.DataArray(fires['acq_date'].values,     dims="fire")

    subset = ds_wind.sel(time=times,
                         lat =latitudes,
                         lon =longitudes,
                         method="nearest")      # <-- nearest cell

    # Define the base names for the eight sectors
    sector_cols = [f"duration_sector_{i}" for i in range(1, 9)]

    # Loop over each sector and assign with optional suffix
    for col in sector_cols:
        fires[f"{col}{suffix}"] = subset[col].values

    return fires
