import numpy as np
import xarray as xr
from .wind_utils import assign_wind_to_sectors

def calculate_wind_speed_direction(ds, ugrd_var="wind_e", vgrd_var="wind_n"):
    ds["wind_speed"] = np.sqrt(ds[ugrd_var]**2 + ds[vgrd_var]**2)
    ds["wind_direction"] = (np.arctan2(ds[vgrd_var], ds[ugrd_var]) * (180 / np.pi)) % 360
    ds["wind_sector"] = xr.DataArray(assign_wind_to_sectors(ds["wind_direction"]), dims=ds["wind_direction"].dims)
    return ds

def calculate_daily_sector_stats(ds):
    daily_data_vars = {}

    for sector in range(1, 9):
        sector_mask = ds['wind_sector'] == sector

        avg_speed = ds['wind_speed'].where(sector_mask).resample(time="1D").mean(dim="time")
        daily_data_vars[f'avg_speed_sector_{sector}'] = avg_speed

        hours_in_sector = ds['wind_sector'].where(sector_mask).resample(time="1D").count(dim="time")
        normalized_duration = hours_in_sector / 24  # Normalize to 24 hours
        daily_data_vars[f'duration_sector_{sector}'] = normalized_duration

    daily_ds = xr.Dataset(
        daily_data_vars,
        coords={"lat": ds.lat, "lon": ds.lon, "time": ds["time"].resample(time="1D").first()}
    )
    return daily_ds
