import xarray as xr

def access_opendap_subset(dataset_url, start_date, end_date, bbox, variables):
    """
    Access a subset of any dataset via OPeNDAP using xarray.

    Parameters and Returns as in your existing function...
    """
    ds = xr.open_dataset(dataset_url, decode_times=True)
    ds = ds.sel(time=slice(start_date, end_date))

    min_lon, min_lat, max_lon, max_lat = bbox
    ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    ds = ds[variables]

    return ds