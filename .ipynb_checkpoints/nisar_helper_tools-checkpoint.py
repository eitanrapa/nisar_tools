import h5py
import numpy as np
import matplotlib.pyplot as plt
import snaphu
import pygmt
import xarray as xr
from pyproj import Transformer
import rioxarray
import math
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

def _multilook_array(arr, max_x, max_y, L, downsample, convolution):
    """
    Convolve and downsample
    """
    if convolution == "Gaussian":
        smoothed = gaussian_filter(arr, sigma=L, mode='constant', cval=0.0)
    if convolution == "Uniform":
        smoothed = uniform_filter(arr, size=L, mode='constant', cval=0.0)
    if downsample:
        smoothed_truncated = smoothed[:max_y, :max_x]
        smoothed = smoothed_truncated[L//2::L, L//2::L]

    return smoothed

def _calculate_multilooked_interferograms(c1, c2, max_x, max_y, looks, downsample, convolution):
    """
    Algorithm for making interferograms
    """
    
    # Calculate raw components at native full resolution
    raw_interf = c1 * np.conj(c2)
    raw_int1 = np.abs(c1)**2
    raw_int2 = np.abs(c2)**2
    
    # Multilook the complex array
    ml_interf = _multilook_array(arr=raw_interf, max_x=max_x, max_y=max_y, L=looks, downsample=downsample, convolution=convolution)
    
    ml_int1 = _multilook_array(arr=raw_int1, max_x=max_x, max_y=max_y, L=looks, downsample=downsample, convolution=convolution)
    ml_int2 = _multilook_array(arr=raw_int2, max_x=max_x, max_y=max_y, L=looks, downsample=downsample, convolution=convolution)

    # Calculate Coherence Magnitude directly on the new downsampled grid
    ml_corr = np.abs(ml_interf) / (np.sqrt(ml_int1 * ml_int2) + 1e-8)
    
    # Force areas that are completely outside the valid swath back to exactly 0.0
    ml_corr = np.nan_to_num(ml_corr, nan=0.0)
    ml_corr = np.clip(ml_corr, 0.0, 1.0)
    
    return ml_interf, ml_corr.astype(np.float32)
    
def calculate_multilooked_interferograms(pair_indices, merged_slcs, x_coords, y_coords, looks=5, downsample=False, convolution="Uniform", processors=1):
    """
    Create interferograms from given pairs
    Pairs must all be on the same x_coords and y_coords 
    """

    # Ensure numpy
    merged_slcs = np.asanyarray(merged_slcs)

    # Calculate maximum truncation limits
    max_y = len(y_coords) // looks * looks
    max_x = len(x_coords) // looks * looks

    if convolution not in ("Uniform", "Gaussian"):
        raise ValueError("convolution must be Uniform or Gaussian")

    if not isinstance(downsample, bool):
        raise ValueError("downsample must be True or False")

    shapes = {slc.shape for slc in merged_slcs}
    if len(shapes) > 1:
        raise ValueError("All SLCs must be the same shape")

    # ODefine a lightweight wrapper function for the threads to call.
    def worker(pair):
        return _calculate_multilooked_interferograms(
            c1=merged_slcs[pair[0]], 
            c2=merged_slcs[pair[1]], 
            max_x=max_x, max_y=max_y, 
            looks=looks, downsample=downsample, convolution=convolution
        )

    # Use ThreadPoolExecutor for zero-copy shared memory
    if processors == 1:
        results = [worker(pair) for pair in pair_indices]
    else:
        with ThreadPoolExecutor(max_workers=processors) as executor:
            results = list(executor.map(worker, pair_indices))

    # Safely unpack the tuples without corrupting 3D shapes via .T
    interferograms_list, coherences_list = zip(*results)
    
    interferograms = np.stack(interferograms_list, axis=0)
    coherences = np.stack(coherences_list, axis=0)

    if downsample:    
        new_x_coords = x_coords[:max_x].reshape(-1, looks).mean(axis=1)
        new_y_coords = y_coords[:max_y].reshape(-1, looks).mean(axis=1)
    else:
        new_x_coords = x_coords
        new_y_coords = y_coords

    return interferograms, coherences, new_x_coords, new_y_coords

# def _calculate_multilooked_interferograms(c1, c2, max_x, max_y, looks, downsample, convolution):
#     """
#     Algorithm for making interferograms
#     """
    
#     # Calculate raw components at native full resolution
#     raw_interf = c1 * np.conj(c2)
#     raw_int1 = np.abs(c1)**2
#     raw_int2 = np.abs(c2)**2
    
#     # Apply Multilooking to all components
#     ml_interf_real = _multilook_array(arr=np.real(raw_interf), max_x=max_x, max_y=max_y, looks=looks, downsample=downsample, convolution=convolution)
#     ml_interf_imag = _multilook_array(arr=np.imag(raw_interf), max_x=max_x, max_y=max_y, looks=looks, downsample=downsample, convolution=convolution)
#     ml_interf = ml_interf_real + 1j * ml_interf_imag
    
#     ml_int1 = _multilook_array(arr=raw_int1, max_x=max_x, max_y=max_y, looks=looks, downsample=downsample, convolution=convolution)
#     ml_int2 = _multilook_array(arr=raw_int2, max_x=max_x, max_y=max_y, looks=looks, downsample=downsample, convolution=convolution)

#     # Calculate Coherence Magnitude directly on the new downsampled grid
#     ml_corr = np.abs(ml_interf) / (np.sqrt(ml_int1 * ml_int2) + 1e-8)
    
#     # Force areas that are completely outside the valid swath back to exactly 0.0
#     ml_corr = np.nan_to_num(ml_corr, nan=0.0)
#     ml_corr = np.clip(ml_corr, 0.0, 1.0)
    
#     return ml_interf, ml_corr.astype(np.float32)
    
# def calculate_multilooked_interferograms(pair_indices, merged_slcs, x_coords, y_coords, looks=5, downsample=False, convolution="Uniform", processors=1):
#     """
#     Create interferograms from given pairs
#     Pairs must all be on the same x_coords and y_coords 
#     """

#     # Ensure numpy
#     merged_slcs = np.asanyarray(merged_slcs)

#     # Calculate maximum truncation limits
#     max_y = len(y_coords) // looks * looks
#     max_x = len(x_coords) // looks * looks

#     if (convolution != "Uniform" and convolution != "Gaussian"):
#         raise ValueError("convolution must be Uniform or Gaussian")

#     if not isinstance(downsample, bool):
#         raise ValueError("downsample must be True or False")

#     shapes = {slc.shape for slc in merged_slcs}
#     if len(shapes) > 1:
#         raise ValueError("All SLCs must be the same shape")

#     # Multiprocess if specified
#     if processors == 1:
#         interferograms, coherences = np.asarray([_calculate_multilooked_interferograms(c1=merged_slc[pair[0]], c2=merged_slc[pair[1]], max_x=max_x, max_y=max_y, looks=looks, downsample=downsample, convolution=convolution) for pair in pair_indices]).T
#     else:
#         with Pool(processors) as p:
#             interferograms, coherences = np.asarray(p.starmap(_calculate_multilooked_interferograms, [[merged_slc[pair[0]], merged_slc[pair[1]], max_x, max_y, looks, downsample, convolution] for pair in pair_indices])).T

#     if downsample:    
#         new_x_coords = x_coords[:max_x].reshape(-1, looks).mean(axis=1)
#         new_y_coords = y_coords[:max_y].reshape(-1, looks).mean(axis=1)
#     else:
#         new_x_coords = x_coords
#         new_y_coords = y_coords

#     return interferograms, coherences, new_x_coords, new_y_coords
    
def _get_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()

def crop_slcs(slcs, x_coords, y_coords, lon_min, lon_max, lat_min, lat_max, epsg_code, direction):
    """
    Crop a bunch of slcs 
    Must all have the same x_coords, y_coords
    """

    for slc in slcs:
        if not isinstance(slc, h5py.Dataset):
            raise ValueError("SLCs must be h5py datasets")
    
    shapes = {slc.shape for slc in slcs}
    if len(shapes) > 1:
        raise ValueError("All SLCs must be the same shape")

    if (direction != "Ascending" and direction != "Descending"):
        raise ValueError("direction must be Ascending or Descending")
        
    # Set up the transformer from Lat/Lon (EPSG:4326) to the Native GSLC projection
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    
    # Transform the 4 corners of the Lat/Lon box to native X/Y
    corners_lon = [lon_min, lon_max, lon_max, lon_min]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    
    x_corners, y_corners = transformer.transform(corners_lon, corners_lat)
    
    # Get the bounding box in the native X/Y coordinate system
    x_slice_min, x_slice_max = min(x_corners), max(x_corners)
    y_slice_min, y_slice_max = min(y_corners), max(y_corners)

    # Handle the direction
    if direction == 'Descending':
        # Y coordinates are descending
        y_slice = slice(y_slice_max, y_slice_min)
    else:
        # Y coordinates are ascending
        y_slice = slice(y_slice_min, y_slice_max)
    
    x_slice = slice(x_slice_min, x_slice_max)
    
    # Find the integer indices corresponding to your spatial bounds
    idx_x_1 = _get_nearest_idx(array=x_coords, value=x_slice_min)
    idx_x_2 = _get_nearest_idx(array=x_coords, value=x_slice_max)
    
    idx_y_1 = _get_nearest_idx(array=y_coords, value=y_slice_min)
    idx_y_2 = _get_nearest_idx(array=y_coords, value=y_slice_max)
    
    # Sort the indices to ensure the slice goes from small index to large index
    x_idx_start, x_idx_end = min(idx_x_1, idx_x_2), max(idx_x_1, idx_x_2)
    y_idx_start, y_idx_end = min(idx_y_1, idx_y_2), max(idx_y_1, idx_y_2)
    
    # Subset the coordinate arrays using the exact same indices
    x_coords_subset = x_coords[x_idx_start:x_idx_end]
    y_coords_subset = y_coords[y_idx_start:y_idx_end]

    cropped_slcs = np.empty((len(slcs), y_idx_end - y_idx_start, x_idx_end - x_idx_start), dtype=np.complex64)
    
    # Read directly from disk into the pre-allocated array slices
    for i, ds in enumerate(slcs):
        cropped_slcs[i] = ds[y_idx_start:y_idx_end, x_idx_start:x_idx_end]
    
    return cropped_slcs, x_coords_subset, y_coords_subset

def merge_rasters(slcs1, slcs2, x_coords1, y_coords1, x_coords2, y_coords2):
    """
    Only merge SLCs in the same track. To merge from different track, merge individually unwrapped interferograms
    """

    # Ensure numpy
    slcs1 = np.asanyarray(slcs1)
    slcs2 = np.asanyarray(slcs2)

    # Check if same amount of slcs
    if not (len(slcs1) == len(slcs2)):
        raise ValueError("Must be same amount of rasters in each list")
    
    # Check if slcs1 all the same x_coords and y_coords
    shapes = {np.shape(slc) for slc in slcs1}
    if len(shapes) > 1:
        raise ValueError("All rasters must be on the same coordinates")

    # Check if slcs1 all the same x_coords and y_coords
    shapes = {np.shape(slc) for slc in slcs2}
    if len(shapes) > 1:
        raise ValueError("All rasters must be on the same coordinates")

    z_dim = np.arange(len(slcs1))

    # Wrap the entire 3D arrays into xarray once
    da1 = xr.DataArray(slcs1, coords={"z": z_dim, "y": y_coords1, "x": x_coords1}, dims=["z", "y", "x"])
    da2 = xr.DataArray(slcs2, coords={"z": z_dim, "y": y_coords1, "x": x_coords2}, dims=["z", "y", "x"])

    # Combine the entire stack in one highly optimized step
    merged_da = da1.combine_first(da2)

    # Extract the raw numpy array and coordinates
    return merged_da.values, merged_da.x.values, merged_da.y.values

def _calculate_snaphu_params(igram_shape, nproc, overlap_target=256):
    rows, cols = igram_shape
    aspect_ratio = rows / cols
    
    # Calculate ideal number of columns based on nproc and aspect ratio
    # cols_tiles^2 * aspect_ratio = nproc
    tiles_col = max(1, int(round(math.sqrt(nproc / aspect_ratio))))
    tiles_row = max(1, int(round(tiles_col * aspect_ratio)))
    
    # Ensure total tiles >= nproc so no processor sits idle
    while (tiles_row * tiles_col) < nproc:
        if (tiles_row / tiles_col) < aspect_ratio:
            tiles_row += 1
        else:
            tiles_col += 1
            
    ntiles = (tiles_row, tiles_col)
    
    # Calculate physical tile sizes
    tile_h = rows // tiles_row
    tile_w = cols // tiles_col
    
    # Ensure overlap isn't taking up more than 25% of the tile size
    max_overlap = int(min(tile_h, tile_w) * 0.25)
    
    # Final overlap is the standard target, capped by the max_overlap limit
    tile_overlap = min(overlap_target, max_overlap)
    
    # Hard floor just in case of tiny arrays
    tile_overlap = max(10, tile_overlap) 

    return ntiles, tile_overlap

def _calculate_snaphu_nlooks(looks_az, looks_rg, spacing_az, spacing_rg, res_az, res_rg):
    """
    From SNAPHU-py
    An estimate of the equivalent number of independent looks may be obtained by


    .. math:: n_e = k_r k_a \frac{d_r d_a}{\rho_r \rho_a}


    where :math:`k_r` and :math:`k_a` are the number of looks in range and azimuth,

    :math:`d_r` and :math:`d_a` are the (single-look) sample spacing in range and

    azimuth, and :math:`\rho_r` and :math:`\rho_a` are the range and azimuth resolution. 
    """

    n_e = np.abs((looks_az * looks_rg) * (spacing_az / res_az) * (spacing_rg / res_rg))
    
    # SNAPHU requires an integer for nlooks, so we round to the nearest whole number
    return round(n_e)

def unwrap_interferograms(igrams, corrs, nlooks, spacing_az, spacing_rg, res_az=8, res_rg=3, nproc=1):
    """
    Unwrap interferograms with SNAPHU
    """
    
    if len(igrams) != len(corrs):
        raise ValueError("Must have the same number of interferograms and coherences")

    shapes = {np.shape(igram) for igram in igrams}
    if len(shapes) > 1:
        raise ValueError("All interferograms must be the same shape")

    ntiles, overlap = _calculate_snaphu_params(igram_shape=np.shape(igrams[0]), nproc=nproc)

    # NISAR resolution params
    nlooks = _calculate_snaphu_nlooks(looks_az=nlooks, looks_rg=nlooks, spacing_az=spacing_az, spacing_rg=spacing_rg, res_az=8, res_rg=3)
        
    unws, conncomps = np.asarray([snaphu.unwrap(igram, corr, nlooks=nlooks, ntiles=ntiles, tile_overlap=overlap, nproc=nproc) for igram, corr in zip(igrams, corrs)]).T
    return unws, conncomps

def calculate_closure_phase():
    return None

def calculate_closure_phase_triplets():
    return None

def mask_water(data, x_coords, y_coords, epsg_code):
    """
    Mask the data
    """
    
    # Calculate grid bounds and spacing
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_spacing = np.abs(x_coords[1] - x_coords[0])
    y_spacing = np.abs(y_coords[1] - y_coords[0])
    
    # Convert bounding box to Lat/Lon for GMT mask generation
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(x_min, y_min)
    lon_max, lat_max = transformer.transform(x_max, y_max)
    
    # Add a tiny buffer to the lat/lon bounding box to ensure full coverage
    buffer = 0.05 
    region_latlon = [lon_min - buffer, lon_max + buffer, lat_min - buffer, lat_max + buffer]
    
    mask_latlon = pygmt.grdlandmask(
        region=region_latlon,
        spacing="5e", 
        mask_values=[np.nan, 1, np.nan, 1, np.nan],
        resolution="f", 
        registration="p" 
    )
        
    # Tell the xarray object that it is currently in Lat/Lon (EPSG:4326)
    mask_latlon = mask_latlon.rio.write_crs("EPSG:4326")
    
    # Reproject it to the NISAR native EPSG
    mask_xy = mask_latlon.rio.reproject(
        f"EPSG:{epsg_code}",
        resolution=(x_spacing, y_spacing)
    )

    data_xr = xr.DataArray(data, coords={"y": y_coords, "x": x_coords}, dims=["y", "x"])

    mask_xy_aligned = mask_xy.interp_like(data_xr, method="nearest")

    return data * mask_xy_aligned

def project_to_latlon(data, x_coords, y_coords, epsg_code):
    """
    Project data to latlon
    """
    
    data_array = xr.DataArray(data, coords={"y": y_coords, "x": x_coords}, dims=["y", "x"])
    
    # Convert Axes to Latitude / Longitude ---
    data_array = data_array.rio.write_crs(f"EPSG:{epsg_code}")
    
    # Reproject the grid to Lat/Lon (Geographic EPSG:4326)
    data_latlon = data_array.rio.reproject("EPSG:4326")

    # Extract 1D numpy arrays
    lats = data_latlon.y.values
    lons = data_latlon.x.values
    
    return data_latlon

def plot_wrapped_interferogram_phase(igram, x_coords, y_coords, epsg_code):
    """
    Plot the wrapped interferogram phases
    """

    phase = np.angle(igram)

    phase_array = xr.DataArray(phase, coords={"y": y_coords, "x": x_coords}, dims=["y", "x"])
    
    # Convert Axes to Latitude / Longitude ---
    # Tell the DataArray what its current native projection is
    phase_array = phase_array.rio.write_crs(f"EPSG:{epsg_code}")
    
    # Reproject the grid to Lat/Lon (Geographic EPSG:4326)
    phase_array_latlon = phase_array.rio.reproject("EPSG:4326")
    
    # --- 3. Make a Nice Phase Plot ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot using a cyclic colormap and exact pi limits
    phase_array_latlon.plot.imshow(
        ax=ax,
        cmap='hsv',       # 'twilight' or 'hsv' are best for cyclic phase
        vmin=-np.pi,           # Lock minimum to -pi
        vmax=np.pi,            # Lock maximum to +pi
        cbar_kwargs={'label': 'Phase (Radians)', 'shrink': 0.8}
    )
    
    # Formatting
    ax.set_title("Wrapped Phase", fontsize=14, pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # Add light gridlines for geographic context
    ax.grid(color='gray', linestyle='--', alpha=0.5)
    
    return fig, ax

def plot_unwrapped_interferogram_phase(unw, x_coords, y_coords, epsg_code):
    """
    Plot the unwrapped interferogram phases
    """

    phase_array = xr.DataArray(unw, coords={"y": y_coords, "x": x_coords}, dims=["y", "x"])
    
    # Convert Axes to Latitude / Longitude ---
    # Tell the DataArray what its current native projection is
    phase_array = phase_array.rio.write_crs(f"EPSG:{epsg_code}")
    
    # Reproject the grid to Lat/Lon (Geographic EPSG:4326)
    phase_array_latlon = phase_array.rio.reproject("EPSG:4326")
    
    # --- 3. Make a Nice Phase Plot ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot using a cyclic colormap and exact pi limits
    phase_array_latlon.plot.imshow(
        ax=ax,
        cmap='RdBu_r',       # 'twilight' or 'hsv' are best for cyclic phase
        cbar_kwargs={'label': 'Phase (Radians)', 'shrink': 0.8}
    )
    
    # Formatting
    ax.set_title("Unwrapped Phase", fontsize=14, pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # Add light gridlines for geographic context
    ax.grid(color='gray', linestyle='--', alpha=0.5)
    
    return fig, ax