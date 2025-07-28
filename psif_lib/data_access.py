import glob
import os
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


def combine_netcdf_files(origin_folder):
    """
    Combines all NetCDF files from a specified origin folder into a single Xarray DataSet.

    Note: We use combine='by_coords', which is excellent for combining files that 
    form a contiguous block in coordinate space (e.g., sequential time slices, or spatial 
    tiles that perfectly abut). Ensure your individual NetCDF files in the origin folder
    have matched latitude and longitude and non-overlapping time dimensions for optimal results.

    Args:
        origin_folder (str): The path to the folder containing the NetCDF files.

    Returns:
        xr.Dataset or None: An xarray Dataset containing the combined data, or None if
                            no NetCDF files are found or an error occurs during combining.
    """

    # Use glob to get all NetCDF files in the origin folder
    file_list = sorted(glob.glob(os.path.join(origin_folder, '*.nc')))

    # Print the list of files that will be combined (for verification)
    print(f"Found {len(file_list)} NetCDF files to combine in '{origin_folder}':")
    for file in file_list:
        print(f"  - {os.path.basename(file)}")

    if len(file_list) > 0:
        try:
            # Combine all files into a single dataset
            # 'combine=by_coords' is optimal for files with sequential time coordinates
            combined_ds = xr.open_mfdataset(file_list, combine='by_coords')

            # Print information about the combined dataset
            print("\nCombined dataset information:")
            print(f"Dimensions: {dict(combined_ds.dims)}")
            if 'time' in combined_ds.coords:
                print(f"Time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")
            else:
                print("No 'time' coordinate found in the combined dataset.")

            return combined_ds

        except Exception as e:
            print(f"An error occurred during combining: {e}")
            return None # Return None if an error occurs
    else:
        print(f"No NetCDF files found in the specified folder: {origin_folder}")
        return None # Return None if no files are found
