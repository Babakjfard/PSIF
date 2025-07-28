
# %%
import os
import pandas as pd
import xarray as xr
from psif_lib import data_access
import psif_lib.wind_utils as wu
import psif_lib.processing as pr
import psif_lib.geo_helpers as geo

# %%
# %%
winds_path = r'data/winds'
winds = data_access.combine_netcdf_files(winds_path)

# %%
fires_path = r'data/fires/fire_archive_M-C61_581147.csv'
fires = pd.read_csv(fires_path)

# %%
fires_with_wind = wu.get_wind_at_fire(fires=fires, ds_wind=winds)

# %%
# Making sure to include fires in the required time period
fires_with_wind = pr.filter_by_month_day(fires_with_wind,
                                     date_col="acq_date",
                                     start_md="01-29",
                                     end_md="06-01")


# %%
# Get the Block Group Centroids
BG_path = 'data/aux/NE-BlockGroups-Centers_of_population.csv'
BGs = pd.read_csv(BG_path)

# %%
# Creating the GEOID for the BGs
pieces = [
    BGs['STATEFP'].astype(str).str.zfill(2),
    BGs['COUNTYFP'].astype(str).str.zfill(3),   # pad to 3
    BGs['TRACTCE'].astype(str).str.zfill(6),
    BGs['BLKGRPCE'].astype(str).str.zfill(1)
]
BGs["GEOID"] = pd.concat(pieces, axis=1).agg("".join, axis=1)
BGs.drop(columns=['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE'], inplace=True)

# %%
fires_with_wind = pr.prep_fires(fires_with_wind)
BGs.rename(columns={'LATITUDE':'latitude', 'LONGITUDE': 'longitude'}, inplace=True)
BGs_final = pr.prep_bgs(BGs)


pairs = pr.fire_bg_pairs(fires_with_wind, BGs_final)
# adding the wind probabilities at BG
pairs.rename(columns={'latitude_bg': 'latitude', 'longitude_bg': 'longitude'}, inplace=True)
pairs = wu.get_wind_at_fire(winds, pairs, "_bg")

PSIF_BG_level = pr.psif_from_pairs(pairs)

# %%
PSIF_with_moving = pr.calculate_moving_average(PSIF_BG_level, 'PSIF', 'GEOID','01/28', '05/31')

# %%
PSIF_with_moving['PSIF_Total'] = PSIF_with_moving['PSIF']+ PSIF_with_moving['moving_avg']

# %%
PSIF_with_moving.to_csv('data/output/PSIF_BGs_NE.csv')

# %%
# Converting into zcta level
BG_to_ZCTA_map = pd.read_csv('data/aux/Nebraska_BG_zcta_crosswalk.csv')
BG_to_ZCTA_map.drop(index=0, inplace=True)

def format_tract(tract):
    # Remove decimal and pad to 6 digits
    tract_str = str(tract)
    if '.' in tract_str:
        left, right = tract_str.split('.')
        right = right.ljust(2, '0')  # Ensure two digits after decimal
        tract_str = left.zfill(4) + right
    else:
        tract_str = tract_str.zfill(6)
    return tract_str

BG_to_ZCTA_map['tract_str'] = BG_to_ZCTA_map['tract'].apply(format_tract)

# county: pad to 3 digits, blockgroup: as string
BG_to_ZCTA_map['county_str'] = BG_to_ZCTA_map['county'].astype(str).str.zfill(3)
BG_to_ZCTA_map['blockgroup_str'] = BG_to_ZCTA_map['blockgroup'].astype(str)

# Now construct GEOID
BG_to_ZCTA_map['GEOID'] = (
    BG_to_ZCTA_map['county_str'] +
    BG_to_ZCTA_map['tract_str'] +
    BG_to_ZCTA_map['blockgroup_str']
)

# %%
BG_to_ZCTA_map['afact'] = pd.to_numeric(BG_to_ZCTA_map['afact'], errors='coerce')
BG_to_ZCTA_map['afact'] = BG_to_ZCTA_map['afact'].astype('float64')

# %%
# Now you can merge as before
merged = pd.merge(PSIF_with_moving, BG_to_ZCTA_map[['GEOID', 'zcta', 'afact']], on='GEOID', how='left')
merged['weighted_psif'] = merged['PSIF_Total'] * merged['afact']
#merged['weighted_mav'] = merged['moving_avg'] * merged['afact']


zcta_psif = (
    merged.groupby(['zcta', 'acq_date'], as_index=False)
    .agg(
        zcta_psif=('weighted_psif', 'sum')
    ))

# %%
zcta_psif.to_csv('data/output/Nebraska_ZCTA_PSIF.csv', index=False)


