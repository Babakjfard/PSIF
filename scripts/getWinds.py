import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

def access_opendap_subset(dataset_url, start_date, end_date, bbox, variables):
    """
    Access a subset of any dataset via OPeNDAP using xarray.

    Parameters:
    - dataset_url (str): The OPeNDAP URL of the dataset.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat).
    - variables (list): List of variable names to extract.

    Returns:
    - xarray.Dataset: The subset dataset.
    """
    # Open the dataset via OPeNDAP
    ds = xr.open_dataset(dataset_url, decode_times=True)

    # Subset by time
    ds = ds.sel(time=slice(start_date, end_date))

    # Subset by spatial bounding box
    min_lon, min_lat, max_lon, max_lat = bbox
    ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

    # Select specified variables
    ds = ds[variables]

    return ds


def assign_wind_to_sectors(wind_direction):
    """
    Assigns wind direction angles (in degrees) to one of eight compass sectors.

    Wind directions are divided into 8 sectors, each spanning 45 degrees:
      1: [337.5, 360) and [0, 22.5)
      2: [22.5, 67.5)
      3: [67.5, 112.5)
      4: [112.5, 157.5)
      5: [157.5, 202.5)
      6: [202.5, 247.5)
      7: [247.5, 292.5)
      8: [292.5, 337.5)

    Parameters
    ----------
    wind_direction : array-like
        Wind direction(s) in degrees. Can be a scalar or a NumPy array.

    Returns
    -------
    sectors : numpy.ndarray
        Array of sector numbers (1-8) assigned to each wind direction.
    """
    sector_bins = np.array([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    sectors = np.digitize(wind_direction, sector_bins, right=True) + 1
    # Correct sector assignment: values in [337.5, 360] and [0, 22.5) belong to sector 1
    sectors = np.where((wind_direction >= 337.5) | (wind_direction < 22.5), 1, sectors)
    return sectors.astype(np.int8)


########### The download process #########################################
# For this project we need data from 2016 to 2023
# And months of February to May, will get three days extra from each side to 
# later be able to make sure we have data if delay is being concidered an option
# nldas_url = "https://hydro1.gesdisc.eosdis.nasa.gov/dods/NLDAS_FORA0125_H.002" ==> Initial link that stopped working.
nldas_url = "https://hydro1.gesdisc.eosdis.nasa.gov/dods/NLDAS_FORA0125_H.2.0"

boundary_100km = np.array([-105.25, 39.09, -94.10, 43.90]) # Nebraska
boundary_wind = boundary_100km - np.array([0.125, 0.125, -0.125, -0.125])
local_tz = 'America/Chicago'

for year in range(2016, 2024):
    start_date = f"{year}-01-29"
    # I added one more day to the end date here, since while time converting the days are pulled 
    # 6 hours back for CDT local timing. If the location was on the east of UTC the hours would have 
    # been pushed some hours forward, that extra day would have therefore be one day ahead of the start date
    end_date = f"{year}-06-04" 
    bounding_box = tuple(boundary_wind)  # (min_lon, min_lat, max_lon, max_lat)
    variables = ['wind_e', 'wind_n']  # 10-m above ground Zonal and Meridional wind speed
    # In new version ugrd --> wind_e (zonal wind), vgrd10m --> wind_n (meridional wind)
    # Access the dataset using the function
    ds = access_opendap_subset(nldas_url, start_date, end_date, bounding_box, variables)
    print(f"Connection made to NLDAS ==> covering dates {start_date} to {end_date}")

    # Print dataset metadata

    # Wind directions
    # ds = nldas_subset

    # Convert to Pandas Series/DatetimeIndex, apply operations, then assign back
    ds['time'] = pd.to_datetime(ds['time'].values).tz_localize('UTC').tz_convert(local_tz).tz_localize(None)

    print(f"*** Times converted to the local time zone: {local_tz}")
    
    ugrd_var = "wind_e"
    vgrd_var = "wind_n"

    print('*** Calculating Wind speed and direction.....')

    ds["wind_speed"] = np.sqrt(ds[ugrd_var]**2 + ds[vgrd_var]**2)
    ds["wind_direction"] = (np.arctan2(ds[vgrd_var], ds[ugrd_var]) * (180 / np.pi)) % 360
    print('***** Wind speed and direction calculated')


    ds["wind_sector"] = xr.DataArray(assign_wind_to_sectors(ds["wind_direction"]), dims=ds["wind_direction"].dims)
    print('** Sectors assigned to the wind directions')


    # Initialize lists to store results for each wind sector
    daily_data_vars = {}

    for sector in range(1, 9):  # Wind sectors 1 to 8
        # Mask for the current wind sector
        sector_mask = ds['wind_sector'] == sector

        # Calculate daily average wind speed for the sector (only where the mask is True)
        avg_speed = ds['wind_speed'].where(sector_mask).resample(time="1D").mean(dim="time")
        daily_data_vars[f'avg_speed_sector_{sector}'] = avg_speed

        # Calculate normalized duration for the sector (hours in sector / total hours in a day)
        hours_in_sector = ds['wind_sector'].where(sector_mask).resample(time="1D").count(dim="time")
        normalized_duration = hours_in_sector / 24  # Normalize by 24 hours
        daily_data_vars[f'duration_sector_{sector}'] = normalized_duration

    print('** Successfully calculated average daily speed and probabilities in each 8 sectors,')

    # Combine all new variables into a new dataset
    daily_ds = xr.Dataset(daily_data_vars, coords={"lat": ds.lat, "lon": ds.lon, "time": ds["time"].resample(time="1D").first()})
    daily_ds = daily_ds.drop_isel(time=[0, -1]) # Remove the first and last day, since those are not complete days due to time adjustment

    # Save the processed dataset to a new NetCDF file (optional)
    ncd_file = f'data/winds/daily_wind_Nebraska_statistics_{year}.nc'
    daily_ds.to_netcdf(ncd_file)
    print(f'** Successfuly saved daily wind components for {year}-01-29 to {year}-06-03')

    print(f"Processing complete. Daily averages and normalized durations have been calculated for {year}.")
    ds.close()
    daily_ds.close()
    
    print('\n\n\n')