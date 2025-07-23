import numpy as np
from psif_lib.data_access import access_opendap_subset
from psif_lib.processing import calculate_wind_speed_direction, calculate_daily_sector_stats

nldas_url = "https://hydro1.gesdisc.eosdis.nasa.gov/dods/NLDAS_FORA0125_H.2.0"
boundary_100km = np.array([-105.25, 39.09, -94.10, 43.90]) # Nebraska
boundary_wind = boundary_100km - np.array([0.125, 0.125, -0.125, -0.125])

data_path = r'data/daily_wind_Nebraska_statistics_test'

for year in range(2009, 2010):
    start_date = f"{year}-01-29"
    end_date = f"{year}-02-03"
    bounding_box = tuple(boundary_wind)
    variables = ['wind_e', 'wind_n']

    ds = access_opendap_subset(nldas_url, start_date, end_date, bounding_box, variables)
    print(f"Connection made to NLDAS ==> covering dates {start_date} to {end_date}")

    ds = calculate_wind_speed_direction(ds)
    print('Wind speed and direction calculated')

    daily_ds = calculate_daily_sector_stats(ds)
    print('Average daily speed and sector durations calculated')

    ncd_file = f'{data_path}_{year}.nc'
    daily_ds.to_netcdf(ncd_file)
    print(f'Saved daily wind components for {start_date} to {end_date}')

print("Processing complete.")
