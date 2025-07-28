import numpy as np
import pandas as pd
import xarray as xr
from .wind_utils import assign_wind_to_sectors
from . import geo_helpers as geo

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

def filter_by_month_day(df, date_col, start_md="01-29", end_md="06-01"):
    """
    Keep only the rows whose date_col falls between start_md (inclusive)
    and end_md (exclusive) in *every* calendar year.

    Parameters
    ----------
    df        : pandas.DataFrame
    date_col  : str
        Name of the column containing dates (datetime64 or YYYY‑MM‑DD string).
    start_md  : str, default "01-29"
        Start month‑day in "MM-DD" or "YYYY-MM-DD" form. Inclusive.
    end_md    : str, default "06-01"
        End   month‑day in "MM-DD" or "YYYY-MM-DD" form. **Exclusive**.

    Returns
    -------
    pandas.DataFrame  (view, not a copy)
    """

    # ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()                          # avoid mutating caller if needed
        df[date_col] = pd.to_datetime(df[date_col])

    # extract month‑day as zero‑padded strings "MM‑DD"
    md = df[date_col].dt.strftime("%m-%d")

    # build boolean mask
    mask = (md >= start_md[-5:]) & (md < end_md[-5:])

    return df.loc[mask]

def prep_fires(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, create sequential *fire_id*, ensure dtypes."""
    out = (df.copy()
              .sort_values("acq_date", kind="mergesort")
              .reset_index(drop=True))
    out["fire_id"] = out.index.astype("int32")
    return out.set_index("fire_id")


def prep_bgs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure GEOID is string and lat/lon numeric."""
    out = df.copy()
    out["GEOID"] = out["GEOID"].astype(str)
    return out

def fire_bg_pairs(fires_d: pd.DataFrame, bgs_d: pd.DataFrame,
                           max_km: float=100) -> pd.DataFrame:
    """
    Create pairs of origins (fire events) and destinatins (block groups centroids) within a given maximum distance.

    This function takes two dataframes containing fire event(origin) locations and destination centroids,
    computes the Cartesian product of the two datasets (matching all fires with all destinations),
    and then filters the pairs to keep only those where the geographic distance between the fire 
    and destination is less than or equal to `max_km`.

    Parameters
    ----------
    fires_d : pandas.DataFrame
        DataFrame containing fire event data with at least `latitude` and `longitude` columns.
    bgs_d : pandas.DataFrame
        DataFrame containing destination data with at least `latitude` and `longitude` columns.
    max_km : float, optional
        Maximum distance in kilometers to consider a fire destination pair valid (default is 100).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing pairs of events (fires) and destination centroids that are within `max_km` distance.
        The DataFrame includes columns from both inputs with suffixes `_fire` and `_bg` 
        appended to overlapping columns, plus a `distance_km` column for the computed distance.
    """
    cart = fires_d.assign(key=1).merge(bgs_d.assign(key=1), on="key", suffixes=("_fire", "_bg"))
    cart.drop(columns="key", inplace=True)

    dist = geo.haversine_km(cart["latitude_fire"].to_numpy(), cart["longitude_fire"].to_numpy(),
                        cart["latitude_bg"].to_numpy(),   cart["longitude_bg"].to_numpy())
    cart = cart.loc[dist <= max_km].copy()
    cart["distance_km"] = dist[dist <= max_km]
    return cart


def psif_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute PSIF contribution per (fire_id,GEOID) and aggregate to BG."""
    # Bearing sector
    sector = assign_wind_to_sectors(
        geo.bearing_deg(pairs["latitude_fire"], pairs["longitude_fire"],
                     pairs["latitude"],   pairs["longitude"]))
    pairs["bearing_sector"] = sector
    # This will turn each element x in `sector` into "duration_sector_" + str(x):
    col_names = "duration_sector_" + pairs["bearing_sector"].astype(str)
    # Now `col_names` is a pandas Series of strings like "duration_sector_3", etc.
    # We can get column indices the same way as before:
    col_positions = pairs.columns.get_indexer(col_names)
    # And finally grab the [row, col] values into a new column:
    pairs["sector_prob"] = pd.to_numeric(pairs.to_numpy()[np.arange(len(pairs)), col_positions])
    
    # 1. prob of the similar direction wind at BG
    col_names_BGWind_SameDirection = "duration_sector_" + pairs["bearing_sector"].astype(str)+"_bg"
    
    # Find columns that actually exist in the DataFrame
    existing_columns = [col for col in col_names_BGWind_SameDirection if col in pairs.columns]
    
    # Get index positions for the existing columns only
    if existing_columns:
        col_position_BGWind_SameDirection = pairs.columns.get_indexer(existing_columns)
        pairs["sector_prob_BGWind_SameDirection"] = pd.to_numeric(pairs.to_numpy()[np.arange(len(pairs)), col_position_BGWind_SameDirection])
    else:
        print("No matching columns found")
    
    # 2. prob of the opposite direction wind at BG
    pairs["bearing_sector_opposite"] = ((pairs["bearing_sector"] + 3) % 8) + 1
    col_names_BGWind_OppositeDirection = "duration_sector_" + pairs["bearing_sector_opposite"].astype(str)+"_bg"
    
    existing_columns_2 = [col for col in col_names_BGWind_OppositeDirection if col in pairs.columns]
    
    if existing_columns_2:
        col_position_BGWind_OppositeDirection = pairs.columns.get_indexer(col_names_BGWind_OppositeDirection)
        pairs["sector_prob_BGWind_OppositeDirection"] = pd.to_numeric(pairs.to_numpy()[np.arange(len(pairs)), col_position_BGWind_OppositeDirection])
    else:
        print("No Matching Column")
    
    # PSIF contribution
    pairs['sector_prob_total'] = (pairs["sector_prob"].astype(float)+ pairs["sector_prob_BGWind_SameDirection"]) # - pairs["sector_prob_BGWind_OppositeDirection"] # NOTE: Removed this to be totally similar to the reference paper
    pairs['sector_prob_total'] = pairs['sector_prob_total'].clip(lower=0)
    pairs["psif_part"] = pairs["frp"].astype(float) *  pairs['sector_prob_total'] / (pairs["distance_km"]**2)
    
    # Aggregate per BG & date
    return (pairs.groupby(["acq_date", "GEOID"], sort=False)["psif_part"].sum()
             .rename("PSIF").reset_index())


def calculate_moving_average(df, value_col, group_col, start_date_str, end_date_str):
    """
    Calculate 3-day moving average for the previous 3 days (excluding current date).
    Compatible with pandas 2.1.4.
    
    Parameters:
    df: pandas DataFrame with date, group, and value columns
    value_col: string, name of the column containing values (e.g., 'PSIF')
    group_col: string, name of the column for grouping (e.g., 'GEOID')
    start_date_str: string, start date in 'MM/dd' format (e.g., '01/28')
    end_date_str: string, end date in 'MM/dd' format (e.g., '05/31')
    
    Returns:
    pandas DataFrame with group, date, and moving average columns
    """
    # Convert acq_date to datetime if not already
    df = df.copy()  # Avoid modifying original dataframe
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # Parse start and end dates (MM/dd format)
    start_month, start_day = map(int, start_date_str.split('/'))
    end_month, end_day = map(int, end_date_str.split('/'))
    
    # Get all unique years in the dataset
    years = sorted(df['acq_date'].dt.year.unique())
    
    # Prepare a list to collect results
    results = []
    
    # Process each year
    for year in years:
        # Create start and end dates for this year
        year_start = pd.Timestamp(year=year, month=start_month, day=start_day)
        year_end = pd.Timestamp(year=year, month=end_month, day=end_day)
        
        # Calculate the moving average start date (start_date + 4 days)
        ma_start = year_start + pd.Timedelta(days=4)
        
        # Create complete date range for this year (from start to end)
        all_dates = pd.date_range(start=year_start, end=year_end, freq='D')
        
        # Create date range for moving average calculation (from ma_start to end)
        ma_dates = pd.date_range(start=ma_start, end=year_end, freq='D')
        
        # Filter dataframe for this year's date range
        year_df = df[(df['acq_date'] >= year_start) & (df['acq_date'] <= year_end)].copy()
        
        # Get unique groups for this year
        if year_df.empty:
            continue
            
        unique_groups = year_df[group_col].unique()
        
        # Process each group for this year
        for group in unique_groups:
            group_df = year_df[year_df[group_col] == group].copy()
            
            # Create a complete DataFrame with all dates for this group
            complete_dates_df = pd.DataFrame({'acq_date': all_dates})
            
            # Merge with group data, filling missing values with 0
            merged_df = pd.merge(complete_dates_df, group_df[['acq_date', value_col]], 
                               on='acq_date', how='left')
            merged_df[value_col] = merged_df[value_col].fillna(0)
            
            # Ensure proper data types (pandas 2.1.4 compatibility)
            merged_df[value_col] = merged_df[value_col].astype('float64')
            
            # Sort by date to ensure proper chronological order
            merged_df = merged_df.sort_values('acq_date').reset_index(drop=True)
            
            # Calculate the 3-day moving average for the previous 3 days (excluding current day)
            merged_df['moving_avg'] = merged_df[value_col].rolling(
                window=3, min_periods=1, center=False
            ).mean().shift(1)
            
            # Fill NaN values in moving_avg with 0
            merged_df['moving_avg'] = merged_df['moving_avg'].fillna(0)
            
            # Filter to only include dates from ma_start to year_end
            result_df = merged_df[merged_df['acq_date'].isin(ma_dates)].copy()
            
            # Add the group column
            result_df[group_col] = group
            
            # Select only required columns
            # result_df = result_df[[group_col, 'acq_date', 'moving_avg']]
            
            # Append to results
            results.append(result_df)
    
    # Concatenate all results
    if results:
        final_result = pd.concat(results, ignore_index=True)
        # Ensure proper sorting
        final_result = final_result.sort_values([group_col, 'acq_date']).reset_index(drop=True)
        return final_result
    else:
        return pd.DataFrame(columns=[group_col, value_col, 'acq_date', 'moving_avg'])

