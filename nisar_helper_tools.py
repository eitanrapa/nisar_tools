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

def multilook_array(arr, max_x, max_y, L, downsample, convolution):
    if convolution == "Gaussian":
        smoothed = gaussian_filter(arr, sigma=L, mode='constant', cval=0.0)
    if convolution == "Uniform":
        smoothed = uniform_filter(arr, size=L, mode='constant', cval=0.0)
    if downsample:
        smoothed_truncated = smoothed[:max_y, :max_x]
        smoothed = smoothed_truncated[L//2::L, L//2::L]

    return smoothed
    
def calculate_multilooked_interferogram(c1, c2, x_coords, y_coords, looks=5, downsample=False, convolution="Uniform"):
    """
    c1 and c2 must be on same grid
    """

    if (convolution != "Uniform" and convolution != "Gaussian"):
        raise ValueError("convolution must be Uniform or Gaussian")

    if not isinstance(downsample, bool):
        raise ValueError("downsample must be True or False")
    
    # Calculate raw components at native full resolution
    raw_interf = c1 * np.conj(c2)
    raw_int1 = np.abs(c1)**2
    raw_int2 = np.abs(c2)**2

    max_y = len(y_coords) // looks * looks
    max_x = len(x_coords) // looks * looks
    
    # Apply Multilooking to all components
    ml_interf_real = multilook_array(np.real(raw_interf), max_x, max_y, looks, downsample, convolution)
    ml_interf_imag = multilook_array(np.imag(raw_interf), max_x, max_y, looks, downsample, convolution)
    ml_interf = ml_interf_real + 1j * ml_interf_imag
    
    ml_int1 = multilook_array(raw_int1, max_x, max_y, looks, downsample, convolution)
    ml_int2 = multilook_array(raw_int2, max_x, max_y, looks, downsample, convolution)

    if downsample:
        
        new_x_coords = x_coords[:max_x].reshape(-1, looks).mean(axis=1)
        new_y_coords = y_coords[:max_y].reshape(-1, looks).mean(axis=1)
    else:
        new_x_coords = x_coords
        new_y_coords = y_coords

    # Calculate Coherence Magnitude directly on the new downsampled grid
    ml_corr = np.abs(ml_interf) / (np.sqrt(ml_int1 * ml_int2) + 1e-8)
    
    # Force areas that are completely outside the valid swath back to exactly 0.0
    ml_corr = np.nan_to_num(ml_corr, nan=0.0)
    ml_corr = np.clip(ml_corr, 0.0, 1.0)
    
    return ml_interf, ml_corr.astype(np.float32), new_x_coords, new_y_coords

# Helper function to find the nearest integer index in a 1D coordinate array
def get_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()

def crop_slc(slc, x_coords, y_coords, lon_min, lon_max, lat_min, lat_max, epsg_code, direction):
    """
    """

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
    idx_x_1 = get_nearest_idx(x_coords, x_slice_min)
    idx_x_2 = get_nearest_idx(x_coords, x_slice_max)
    
    idx_y_1 = get_nearest_idx(y_coords, y_slice_min)
    idx_y_2 = get_nearest_idx(y_coords, y_slice_max)
    
    # Sort the indices to ensure the slice goes from small index to large index
    x_idx_start, x_idx_end = min(idx_x_1, idx_x_2), max(idx_x_1, idx_x_2)
    y_idx_start, y_idx_end = min(idx_y_1, idx_y_2), max(idx_y_1, idx_y_2)
    
    # Subset the coordinate arrays using the exact same indices
    x_coords_subset = x_coords[x_idx_start:x_idx_end]
    y_coords_subset = y_coords[y_idx_start:y_idx_end]

    slc_subset = slc[y_idx_start:y_idx_end, x_idx_start:x_idx_end]
    
    return slc_subset, x_coords_subset, y_coords_subset

def merge_raster(raster1, raster2, x_coords1, y_coords1, x_coords2, y_coords2):
    """
    Only merge SLCs in the same track. To merge from different track, merge individually unwrapped interferograms
    """
    raster1_array = xr.DataArray(raster1, coords={"y": y_coords1, "x": x_coords1}, dims=["y", "x"])
    raster2_array = xr.DataArray(raster2, coords={"y": y_coords1, "x": x_coords2}, dims=["y", "x"])
    merged_raster = raster1_array.combine_first(raster2_array)

    return merged_raster, merged_raster.x.values, merged_raster.y.values

def calculate_snaphu_params(igram_shape, nproc, overlap_target=256):
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

def calculate_snaphu_nlooks(looks_az, looks_rg, spacing_az, spacing_rg, res_az, res_rg):
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

def unwrap_interferogram(igram, corr, nlooks, spacing_az, spacing_rg, res_az=8, res_rg=3, nproc=1):

    ntiles, overlap = calculate_snaphu_params(igram_shape=np.shape(igram), nproc=nproc)

    # NISAR resolution params
    nlooks = calculate_snaphu_nlooks(looks_az=nlooks, looks_rg=nlooks, spacing_az=spacing_az, 
                                     spacing_rg=spacing_rg, res_az=8,  res_rg=3)

    unw, conncomp = snaphu.unwrap(igram, corr, nlooks=nlooks, ntiles=ntiles, tile_overlap=overlap, nproc=nproc)
    return unw, conncomp

def mask_water(data, x_coords, y_coords, epsg_code):
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

    data_xr = xr.DataArray(
    data,
    coords={"y": y_coords, "x": x_coords},
    dims=["y", "x"])

    mask_xy_aligned = mask_xy.interp_like(data_xr, method="nearest")

    return data * mask_xy_aligned

def project_to_latlon(data, x_coords, y_coords, epsg_code):
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