#!/usr/bin/env python3
"""
PyCASSO FITS File to TSV conversion. Extraction of line-of-sight 
velocity (V_0)and velocity dispersion (V_D) and error calculation 
for spaxels/points along the major axis baseed on CALIFA position angle.

Author: Marina M. Alioto
Date: 6 Oct, 2025
"""

print("SCRIPT STARTED - If you see this, Python is working")

import sys
print("sys imported")
import numpy as np
print("numpy imported")
from astropy.io import fits
print("astropy imported")
import matplotlib.pyplot as plt
print("matplotlib imported")
import pandas as pd
print("pandas imported")
import os
from pathlib import Path

def calculate_velocity_error(sn_zone):
    """
    Calculate velocity error based on signal-to-noise ratio.
    From de Amorim, et al Table 3.
    
    Parameters:
    -----------
    sn_zone : float or array
        Signal-to-noise ratio from SN_ZONE field
        
    Returns:
    --------
    v_err : float or array
        Velocity error in km/s
        - 19 km/s at SN=20
        - 9 km/s at SN=50
        - 9 km/s for SN>50
        - Linear interpolation between 20 and 50
    """
    sn = np.asarray(sn_zone)
    v_err = np.zeros_like(sn, dtype=float)
    
    # Below SN=20, set to max error (19 km/s)
    v_err[sn <= 20] = 19.0
    
    # Between SN=20 and SN=50, linear interpolation
    mask_interp = (sn > 20) & (sn < 50)
    v_err[mask_interp] = 19.0 - (19.0 - 9.0) * (sn[mask_interp] - 20.0) / (50.0 - 20.0)
    
    # Above SN=50, set to min error (9 km/s)
    v_err[sn >= 50] = 9.0
    
    return v_err

def calculate_dispersion_error(sn_zone):
    """
    Calculate velocity dispersion error based on signal-to-noise ratio.
    From de Amorim, et al Table 3.

    Parameters:
    -----------
    sn_zone : float or array
        Signal-to-noise ratio from SN_ZONE field
        
    Returns:
    --------
    vd_err : float or array
        Velocity dispersion error in km/s
        - 22 km/s at SN=20
        - 10 km/s at SN=50
        - 10 km/s for SN>50
        - Linear interpolation between 20 and 50
    """
    sn = np.asarray(sn_zone)
    vd_err = np.zeros_like(sn, dtype=float)
    
    # Below SN=20, set to max error (22 km/s)
    vd_err[sn <= 20] = 22.0
    
    # Between SN=20 and SN=50, linear interpolation
    mask_interp = (sn > 20) & (sn < 50)
    vd_err[mask_interp] = 22.0 - (22.0 - 10.0) * (sn[mask_interp] - 20.0) / (50.0 - 20.0)
    
    # Above SN=50, set to min error (10 km/s)
    vd_err[sn >= 50] = 10.0
    
    return vd_err

def read_califa_photometric_catalog(catalog_path="/Users/talioto/Documents/Marina/astrophysics/califa/dr3/CALIFA_8_MS_SDSS_mag.csv"):
    """
    Read the CALIFA photometric catalog and return a dictionary keyed by CALIFA ID.
    """
    try:
        print(f"INFO: Reading CALIFA photometric catalog: {catalog_path}")
        
        # Read the CSV file, skipping comment lines
        califa_data = {}
        
        with open(catalog_path, 'r') as f:
            lines = f.readlines()
        
        # Find the start of data (skip header comments)
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip():
                data_start = i
                break
        
        print(f"INFO: Found data starting at line {data_start + 1}")
        
        # Parse data lines
        for line_num, line in enumerate(lines[data_start:], start=data_start + 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                # Split by comma and clean up whitespace
                parts = [part.strip() for part in line.split(',')]
                
                if len(parts) >= 17:  # Ensure we have enough columns
                    califa_id = int(parts[0])
                    pa_align = float(parts[1])  # degrees from North counterclockwise
                    ba = float(parts[2])  # axis ratio b/a
                    el_hlr = float(parts[19])  # half-light radius in arcsec
                    #el_hlr = float(parts[14])  # half-light radius in arcsec
                    
                    califa_data[califa_id] = {
                        'pa_align': pa_align,
                        'ba': ba,
                        'el_hlr': el_hlr,
                        'line_number': line_num
                    }
                    
                    if len(califa_data) <= 5:  # Show first few entries
                        print(f"INFO: CALIFA {califa_id}: PA={pa_align:.1f}°, b/a={ba:.3f}, R_eff={el_hlr:.1f}\"")
                        
            except (ValueError, IndexError) as e:
                print(f"WARNING: Could not parse line {line_num}: {e}")
                continue
        
        print(f"SUCCESS: Loaded {len(califa_data)} galaxies from CALIFA catalog")
        return califa_data
        
    except FileNotFoundError:
        print(f"WARNING: CALIFA catalog not found at {catalog_path}")
        return {}
    except Exception as e:
        print(f"ERROR reading CALIFA catalog: {e}")
        return {}

def extract_califa_id_from_filename(filename):
    """
    Extract CALIFA ID from filename. Common patterns:
    - K0001_*.fits → CALIFA ID 1
    - NGC0001_*.fits → look up in catalog by name
    - califa_001_*.fits → CALIFA ID 1
    """
    import re
    from pathlib import Path
    
    basename = Path(filename).stem.upper()
    print(f"INFO: Extracting CALIFA ID from filename: {basename}")
    
    # Pattern 1: K followed by digits (K0001 → 1)
    match = re.search(r'K0*(\d+)', basename)
    if match:
        califa_id = int(match.group(1))
        print(f"INFO: Found CALIFA ID from K pattern: {califa_id}")
        return califa_id
    
    # Pattern 2: Direct CALIFA ID pattern
    match = re.search(r'CALIFA[_-]0*(\d+)', basename)
    if match:
        califa_id = int(match.group(1))
        print(f"INFO: Found CALIFA ID from CALIFA pattern: {califa_id}")
        return califa_id
    
    # Pattern 3: Just numbers (try to extract first number group)
    match = re.search(r'0*(\d+)', basename)
    if match:
        califa_id = int(match.group(1))
        print(f"INFO: Found potential CALIFA ID from number pattern: {califa_id}")
        return califa_id
    
    print(f"WARNING: Could not extract CALIFA ID from filename: {basename}")
    return None

def get_califa_photometric_data(filename, califa_catalog):
    """
    Get photometric data for a galaxy from the CALIFA catalog.
    """
    if not califa_catalog:
        print("INFO: No CALIFA catalog available")
        return {}
    
    califa_id = extract_califa_id_from_filename(filename)
    if califa_id is None:
        print("WARNING: Could not determine CALIFA ID from filename")
        return {}
    
    if califa_id in califa_catalog:
        data = califa_catalog[califa_id]
        print(f"SUCCESS: Found CALIFA {califa_id} photometric data:")
        print(f"  Position angle: {data['pa_align']:.1f}° (from North, counterclockwise)")
        print(f"  Axis ratio b/a: {data['ba']:.3f}")
        print(f"  Half-light radius: {data['el_hlr']:.1f} arcsec")
        return data
    else:
        print(f"WARNING: CALIFA ID {califa_id} not found in catalog")
        print(f"INFO: Available IDs range from {min(califa_catalog.keys())} to {max(califa_catalog.keys())}")
        return {}

def extract_all_metadata(hdul):
    """
    Extract all metadata from all HDU headers and organize it.
    Returns a dictionary with comprehensive metadata information.
    """
    metadata = {}
    
    print("INFO: Extracting comprehensive metadata from all HDU headers...")
    
    for i, hdu in enumerate(hdul):
        hdu_name = hdu.name if hasattr(hdu, 'name') and hdu.name else f"HDU{i}"
        print(f"INFO: Processing {hdu_name} header...")
        
        if hasattr(hdu, 'header') and hdu.header:
            hdu_metadata = {}
            for key in hdu.header.keys():
                try:
                    value = hdu.header[key]
                    comment = hdu.header.comments[key] if key in hdu.header.comments else ""
                    hdu_metadata[key] = {
                        'value': value,
                        'comment': comment,
                        'hdu': hdu_name
                    }
                except Exception as e:
                    print(f"WARNING: Could not extract key {key} from {hdu_name}: {e}")
            
            metadata[hdu_name] = hdu_metadata
    
    return metadata

def extract_photometric_info(metadata, califa_data=None):
    """
    Extract photometric information from metadata including effective radius,
    axis ratios, distances, and IFU coverage. Now also includes CALIFA catalog data.
    """
    photometric_info = {}
    
    # Add CALIFA catalog data first (priority over FITS headers)
    if califa_data:
        photometric_info['califa_pa_align'] = califa_data['pa_align']
        photometric_info['califa_ba'] = califa_data['ba']
        photometric_info['califa_el_hlr'] = califa_data['el_hlr']
        print(f"INFO: Added CALIFA catalog data: PA={califa_data['pa_align']:.1f}°, b/a={califa_data['ba']:.3f}")
    
    # Search through all headers for photometric keywords
    all_keys = {}
    for hdu_name, hdu_meta in metadata.items():
        for key, info in hdu_meta.items():
            all_keys[key] = info
    
    # Distance-related keywords
    distance_keywords = ['DIST', 'DISTANCE', 'DIST_MPC', 'D_MPC', 'DL', 'DA', 'REDSHIFT', 'Z']
    for key in distance_keywords:
        if key in all_keys:
            photometric_info[f'distance_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found distance parameter {key}: {all_keys[key]['value']}")
    
    # Axis ratio keywords (CALIFA takes priority)
    axis_keywords = ['BA', 'B_A', 'AXIS_RATIO', 'ELLIP', 'E', 'Q']
    for key in axis_keywords:
        if key in all_keys and 'califa_ba' not in photometric_info:
            photometric_info[f'axis_ratio_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found axis ratio parameter {key}: {all_keys[key]['value']}")
    
    # Effective radius keywords (CALIFA takes priority)
    radius_keywords = ['RE', 'R_E', 'REFF', 'R_EFF', 'HLR', 'HALF_LIGHT', 'RE_ARCSEC', 'RE_KPC']
    for key in radius_keywords:
        if key in all_keys and 'califa_el_hlr' not in photometric_info:
            photometric_info[f'radius_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found radius parameter {key}: {all_keys[key]['value']}")
    
    # Position angle keywords (CALIFA takes priority)
    pa_keywords = ['PA', 'POSANG', 'POS_ANG', 'PA_PHOT', 'PA_DISK']
    for key in pa_keywords:
        if key in all_keys and 'califa_pa_align' not in photometric_info:
            photometric_info[f'pa_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found position angle parameter {key}: {all_keys[key]['value']}")
    
    # IFU/Field of view keywords
    fov_keywords = ['FOV', 'FIELD', 'IFU_SIZE', 'SPAXEL', 'PIXSCALE', 'CDELT1', 'CDELT2', 'PLATE_SCALE']
    for key in fov_keywords:
        if key in all_keys:
            photometric_info[f'fov_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found field of view parameter {key}: {all_keys[key]['value']}")
    
    # Coordinates
    coord_keywords = ['RA', 'DEC', 'CRVAL1', 'CRVAL2', 'OBJRA', 'OBJDEC']
    for key in coord_keywords:
        if key in all_keys:
            photometric_info[f'coord_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found coordinate parameter {key}: {all_keys[key]['value']}")
    
    # Object identification
    id_keywords = ['OBJECT', 'OBJNAME', 'PLATEIFU', 'MANGAID', 'NSA_ID', 'CALIFA_ID']
    for key in id_keywords:
        if key in all_keys:
            photometric_info[f'id_{key.lower()}'] = all_keys[key]['value']
            print(f"INFO: Found object ID parameter {key}: {all_keys[key]['value']}")
    
    return photometric_info, all_keys

def calculate_physical_scales(photometric_info):
    """
    Calculate physical scales from distance and angular information.
    """
    scales = {}
    
    # Find distance in Mpc
    distance_mpc = None
    for key, value in photometric_info.items():
        if 'distance' in key.lower() and isinstance(value, (int, float)):
            if 'mpc' in key.lower() or value > 0.1:  # Assume values > 0.1 are in Mpc
                distance_mpc = float(value)
                scales['distance_mpc'] = distance_mpc
                print(f"INFO: Using distance: {distance_mpc:.2f} Mpc")
                break
    
    # Find pixel scale
    pixel_scale_arcsec = None
    for key, value in photometric_info.items():
        if ('pixscale' in key.lower() or 'cdelt' in key.lower() or 'plate_scale' in key.lower()) and isinstance(value, (int, float)):
            pixel_scale_arcsec = abs(float(value)) * 3600  # Convert degrees to arcsec if needed
            if pixel_scale_arcsec > 10:  # Already in arcsec
                pixel_scale_arcsec = abs(float(value))
            scales['pixel_scale_arcsec'] = pixel_scale_arcsec
            print(f"INFO: Using pixel scale: {pixel_scale_arcsec:.3f} arcsec/pixel")
            break
    
    # Calculate kpc per arcsec
    if distance_mpc is not None:
        kpc_per_arcsec = distance_mpc * 4.848e-3
        scales['kpc_per_arcsec'] = kpc_per_arcsec
        print(f"INFO: Calculated scale: {kpc_per_arcsec:.4f} kpc/arcsec")
        
        if pixel_scale_arcsec is not None:
            kpc_per_pixel = kpc_per_arcsec * pixel_scale_arcsec
            scales['kpc_per_pixel'] = kpc_per_pixel
            print(f"INFO: Calculated scale: {kpc_per_pixel:.4f} kpc/pixel")
    
    # Convert effective radii to different units
    for key, value in photometric_info.items():
        if 'radius' in key.lower() and isinstance(value, (int, float)):
            radius_arcsec = float(value)
            scales[f'{key}_arcsec'] = radius_arcsec
            
            if distance_mpc is not None:
                radius_kpc = radius_arcsec * scales['kpc_per_arcsec']
                scales[f'{key}_kpc'] = radius_kpc
                print(f"INFO: {key}: {radius_arcsec:.2f} arcsec = {radius_kpc:.3f} kpc")
    
    return scales

def extract_all_spaxel_data(hdul):
    """
    Extract data from all spaxels for all extensions with data arrays.
    Returns a comprehensive DataFrame with all spaxel information.
    """
    print("INFO: Extracting all spaxel data from all extensions...")
    
    # Find all extensions with 2D data
    data_extensions = {}
    for i, hdu in enumerate(hdul):
        hdu_name = hdu.name if hasattr(hdu, 'name') and hdu.name else f"HDU{i}"
        if hasattr(hdu, 'data') and hdu.data is not None:
            if len(hdu.data.shape) == 2:  # 2D arrays
                data_extensions[hdu_name] = hdu.data
                print(f"INFO: Found 2D data in {hdu_name} with shape {hdu.data.shape}")
            elif len(hdu.data.shape) == 3:  # 3D arrays - take first slice or handle specially
                data_extensions[hdu_name] = hdu.data[0] if hdu.data.shape[0] == 1 else hdu.data
                print(f"INFO: Found 3D data in {hdu_name} with shape {hdu.data.shape}")
    
    if not data_extensions:
        print("WARNING: No 2D data extensions found!")
        return pd.DataFrame()
    
    # Get dimensions from first data array
    first_ext = list(data_extensions.values())[0]
    if len(first_ext.shape) == 2:
        ny, nx = first_ext.shape
    else:
        print("ERROR: Cannot determine array dimensions")
        return pd.DataFrame()
    
    print(f"INFO: Creating spaxel table for {nx}x{ny} = {nx*ny} spaxels")
    
    # Create coordinate grids
    Y, X = np.indices((ny, nx))  # Y = row, X = column
    
    # Initialize data dictionary
    spaxel_data = {
        'x_pixel': X.flatten(),
        'y_pixel': Y.flatten(),
        'spaxel_id': np.arange(nx * ny)
    }
    
    # Extract data from each extension
    for ext_name, data_array in data_extensions.items():
        if len(data_array.shape) == 2:
            # 2D array - one value per spaxel
            spaxel_data[ext_name] = data_array.flatten()
            print(f"INFO: Added {ext_name} data ({np.sum(np.isfinite(data_array.flatten()))} valid spaxels)")
        elif len(data_array.shape) == 3:
            # 3D array - multiple values per spaxel (e.g., spectral data)
            for i in range(data_array.shape[0]):
                spaxel_data[f'{ext_name}_slice_{i}'] = data_array[i].flatten()
            print(f"INFO: Added {ext_name} data with {data_array.shape[0]} slices")
    
    # Create DataFrame
    df = pd.DataFrame(spaxel_data)
    
    # Add radial distance from center
    cx, cy = nx / 2.0, ny / 2.0
    df['radius_pixels_geometric'] = np.sqrt((df['x_pixel'] - cx)**2 + (df['y_pixel'] - cy)**2)
    
    # === NEW CODE: Add error calculations if SN_ZONE is available ===
    if 'SN_ZONE' in df.columns:
        print("INFO: Calculating velocity and dispersion errors from SN_ZONE")
        df['v_err'] = calculate_velocity_error(df['SN_ZONE'].values)
        df['vd_err'] = calculate_dispersion_error(df['SN_ZONE'].values)
        
        # Print statistics
        valid_sn = df['SN_ZONE'][np.isfinite(df['SN_ZONE'])]
        if len(valid_sn) > 0:
            print(f"INFO: S/N range: {valid_sn.min():.1f} to {valid_sn.max():.1f}")
            print(f"INFO: v_err range: {df['v_err'].min():.1f} to {df['v_err'].max():.1f} km/s")
            print(f"INFO: vd_err range: {df['vd_err'].min():.1f} to {df['vd_err'].max():.1f} km/s")
    else:
        print("WARNING: SN_ZONE not found in data, cannot calculate errors")
        df['v_err'] = np.nan
        df['vd_err'] = np.nan
    
    print(f"INFO: Created spaxel DataFrame with shape {df.shape}")
    print(f"INFO: Columns: {list(df.columns)}")
    
    return df

def find_kinematic_center(velocity_map, sigma_map=None):
    """
    Find kinematic center where velocity dispersion peaks and mean velocity is near zero.
    Returns center as (x_center, y_center) where x=column, y=row for plotting consistency.
    """
    if velocity_map is None:
        print("WARNING: No velocity map provided, returning None")
        return None
    
    ny, nx = velocity_map.shape
    print(f"INFO: Velocity map dimensions: {nx} x {ny} (width x height)")
    
    # Valid pixels are those with finite velocity values
    valid = np.isfinite(velocity_map)
    n_valid = np.sum(valid)
    print(f"INFO: Found {n_valid} valid pixels out of {nx*ny} total pixels")
    
    if n_valid < 10:
        print("WARNING: Too few valid pixels (<10), using geometric center")
        return (nx / 2.0, ny / 2.0)  # Return as (x, y) = (column, row)
    
    # Print velocity statistics
    vel_min = np.nanmin(velocity_map[valid])
    vel_max = np.nanmax(velocity_map[valid])
    vel_mean = np.nanmean(velocity_map[valid])
    vel_median = np.nanmedian(velocity_map[valid])
    print(f"INFO: Velocity range: {vel_min:.1f} to {vel_max:.1f} km/s")
    print(f"INFO: Velocity mean: {vel_mean:.1f} km/s, median: {vel_median:.1f} km/s")
    
    # Create coordinate grids: X[row, col] = col, Y[row, col] = row
    Y, X = np.indices((ny, nx))  # Y = row indices, X = column indices
    print(f"INFO: Coordinate grid ranges: X=[0,{nx-1}], Y=[0,{ny-1}]")
    centers = []  # Store different center estimates
    
    # Method 1: PRIORITIZED - Find where velocity dispersion peaks (galaxy center)
    if sigma_map is not None:
        print("INFO: Method 1 (PRIORITIZED) - Using velocity dispersion peak")
        sigma_valid = np.isfinite(sigma_map)
        
        if np.sum(sigma_valid) > 10:
            # Find peak in velocity dispersion (smooth first to avoid noise)
            try:
                from scipy import ndimage
                # Smooth the sigma map
                sigma_smooth = ndimage.gaussian_filter(sigma_map, sigma=1.0)
                sigma_smooth = np.where(sigma_valid, sigma_smooth, np.nan)
                
                # Find the maximum dispersion location
                max_sigma_idx = np.nanargmax(sigma_smooth)
                max_sigma_y, max_sigma_x = np.unravel_index(max_sigma_idx, sigma_smooth.shape)
                
                print(f"INFO: Peak dispersion found at x={max_sigma_x} (col), y={max_sigma_y} (row)")
                print(f"INFO: Peak dispersion value: {sigma_map[max_sigma_y, max_sigma_x]:.2f} km/s")
                
                # Also check nearby region for consistency
                search_radius = 3
                y_min, y_max = max(0, max_sigma_y - search_radius), min(ny, max_sigma_y + search_radius + 1)
                x_min, x_max = max(0, max_sigma_x - search_radius), min(nx, max_sigma_x + search_radius + 1)
                
                region_mask = sigma_valid[y_min:y_max, x_min:x_max]
                if np.sum(region_mask) > 5:
                    # Weight by sigma values in the peak region
                    Y_region = Y[y_min:y_max, x_min:x_max]
                    X_region = X[y_min:y_max, x_min:x_max]
                    sigma_region = sigma_map[y_min:y_max, x_min:x_max]
                    
                    weights = sigma_region[region_mask]
                    x_center1 = np.average(X_region[region_mask], weights=weights)
                    y_center1 = np.average(Y_region[region_mask], weights=weights)
                    
                    centers.append(('sigma_peak', x_center1, y_center1))
                    print(f"SUCCESS: Method 1 (sigma peak) center: x={x_center1:.1f} (col), y={y_center1:.1f} (row)")
                else:
                    centers.append(('sigma_peak', float(max_sigma_x), float(max_sigma_y)))
                    print(f"SUCCESS: Method 1 (sigma peak) center: x={max_sigma_x:.1f} (col), y={max_sigma_y:.1f} (row)")
                    
            except ImportError:
                print("WARNING: scipy not available for smoothing, using raw sigma peak")
                max_sigma_idx = np.nanargmax(sigma_map)
                max_sigma_y, max_sigma_x = np.unravel_index(max_sigma_idx, sigma_map.shape)
                centers.append(('sigma_peak', float(max_sigma_x), float(max_sigma_y)))
                print(f"SUCCESS: Method 1 (sigma peak) center: x={max_sigma_x:.1f} (col), y={max_sigma_y:.1f} (row)")
        else:
            print("WARNING: Not enough valid sigma pixels")
    else:
        print("INFO: No sigma map provided, skipping sigma peak method")
    
    # Method 2: Find where velocity is closest to zero (systemic velocity)
    print("INFO: Method 2 - Using near-zero velocity")
    systemic_vel = vel_median  # Use median as estimate of systemic velocity
    vel_from_systemic = np.abs(velocity_map - systemic_vel)
    zero_threshold = np.nanpercentile(vel_from_systemic[valid], 15)  # Use 15th percentile for tighter constraint
    near_systemic = valid & (vel_from_systemic <= zero_threshold)
    n_near_systemic = np.sum(near_systemic)
    
    print(f"INFO: Using systemic velocity estimate: {systemic_vel:.1f} km/s")
    print(f"INFO: Near-zero velocity threshold: {zero_threshold:.1f} km/s")
    print(f"INFO: Found {n_near_systemic} pixels near systemic velocity")
    
    if n_near_systemic > 10:
        # Weight by inverse of velocity difference (closer to zero = higher weight)
        weights = 1.0 / (vel_from_systemic[near_systemic] + 0.01)
        
        # Also weight by sigma if available (higher sigma = more weight, as center should have high dispersion)
        if sigma_map is not None:
            sigma_weights = sigma_map[near_systemic]
            sigma_weights = np.where(np.isfinite(sigma_weights), sigma_weights, 0)
            if np.sum(sigma_weights) > 0:
                print("INFO: Also using sigma weighting for zero-velocity method")
                sigma_weights = sigma_weights / np.nanmax(sigma_weights)
                weights = weights * (1 + 2 * sigma_weights)  # Give more weight to sigma
        
        x_center2 = np.average(X[near_systemic], weights=weights)  # column = x
        y_center2 = np.average(Y[near_systemic], weights=weights)  # row = y
        centers.append(('zero_velocity', x_center2, y_center2))
        print(f"INFO: Method 2 (zero velocity) center: x={x_center2:.1f} (col), y={y_center2:.1f} (row)")
    
    # Method 3: Velocity gradient method (as backup)
    print("INFO: Method 3 - Using velocity gradient (backup)")
    gy, gx = np.gradient(velocity_map)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_valid = valid & np.isfinite(grad_mag)
    
    if np.sum(grad_valid) > 10:
        # Find regions with high velocity gradient
        high_grad_threshold = np.nanpercentile(grad_mag[grad_valid], 80)
        high_grad = grad_valid & (grad_mag >= high_grad_threshold)
        
        if np.sum(high_grad) > 5:
            # Weight by gradient magnitude
            weights = grad_mag[high_grad]
            x_center3 = np.average(X[high_grad], weights=weights)  # column = x
            y_center3 = np.average(Y[high_grad], weights=weights)  # row = y
            centers.append(('gradient', x_center3, y_center3))
            print(f"INFO: Method 3 (gradient) center: x={x_center3:.1f} (col), y={y_center3:.1f} (row)")
    
    # Choose the best center - PRIORITIZE SIGMA PEAK, then ZERO VELOCITY
    if len(centers) == 0:
        print("WARNING: No centers found, using geometric center")
        final_center = (nx / 2.0, ny / 2.0)  # (x=col, y=row)
    else:
        print("INFO: Available centers:")
        for method, x, y in centers:
            print(f"  {method}: x={x:.1f} (col), y={y:.1f} (row)")
        
        # PRIORITIZE sigma peak method if available
        sigma_centers = [c for c in centers if c[0] == 'sigma_peak']
        if sigma_centers:
            final_center = (sigma_centers[0][1], sigma_centers[0][2])  # (x, y)
            print("SUCCESS: Choosing SIGMA PEAK method (prioritized)")
        else:
            # Fallback to zero velocity method if available
            zero_centers = [c for c in centers if c[0] == 'zero_velocity']
            if zero_centers:
                final_center = (zero_centers[0][1], zero_centers[0][2])  # (x, y)
                print("INFO: Sigma peak not available, choosing zero velocity method")
            else:
                # Otherwise use gradient method
                gradient_centers = [c for c in centers if c[0] == 'gradient']
                if gradient_centers:
                    final_center = (gradient_centers[0][1], gradient_centers[0][2])  # (x, y)
                    print("INFO: Using gradient method as backup")
                else:
                    final_center = (centers[0][1], centers[0][2])  # (x, y)
                    print(f"INFO: Using fallback method: {centers[0][0]}")
    
    # Compare with geometric center
    geom_center_x, geom_center_y = nx / 2.0, ny / 2.0
    offset_x = final_center[0] - geom_center_x
    offset_y = final_center[1] - geom_center_y
    offset_dist = np.sqrt(offset_x**2 + offset_y**2)
    print(f"INFO: Final center: x={final_center[0]:.1f} (col), y={final_center[1]:.1f} (row)")
    print(f"INFO: Geometric center: x={geom_center_x:.1f} (col), y={geom_center_y:.1f} (row)")
    print(f"INFO: Offset from geometric center: dx={offset_x:.1f}, dy={offset_y:.1f} pixels = {offset_dist:.1f} pixels")
    
    return final_center

def extract_major_axis_profile_advanced(velocity_map, sigma_map, center=None, pa=None, inclination=None,
                                       band_width_arcsec=3.0, pixel_scale_arcsec=None, method='mean',sn_zone_map=None):
    """
    Advanced version with different statistical methods and returns all individual points.
    
    Returns:
    --------
    positions : array
        Position along major axis (pixels from center)  
    velocities : array
        Mean/median/weighted velocity at each position
    sigmas : array  
        Mean dispersion at each position
    all_points_data : dict
        Dictionary containing all individual spaxel data used in the analysis
    """
    ny, nx = velocity_map.shape
    cx, cy = (nx / 2.0, ny / 2.0) if center is None else center
    if pa is None:
        pa = 0.0
    
    # Convert band width to pixels
    if pixel_scale_arcsec is not None:
        band_width_pixels = band_width_arcsec / pixel_scale_arcsec
        print(f"INFO: Advanced extraction - Band width: {band_width_arcsec}\" = {band_width_pixels:.1f} pixels")
    else:
        band_width_pixels = 3.0
        print(f"WARNING: No pixel scale provided, using {band_width_pixels} pixels for band width")
    
    # Convert PA to radians and get direction vectors
    pa_rad = np.radians(90.0 - pa)
    major_cos, major_sin = np.cos(pa_rad), np.sin(pa_rad)
    minor_cos, minor_sin = np.cos(pa_rad + np.pi/2), np.sin(pa_rad + np.pi/2)
    
    print(f"INFO: Major axis direction: ({major_cos:.3f}, {major_sin:.3f})")
    print(f"INFO: Minor axis direction: ({minor_cos:.3f}, {minor_sin:.3f})")
    
    # Create coordinate grids
    Y, X = np.indices((ny, nx))
    
    # Transform to major/minor axis coordinate system
    # Distance along major axis from center
    major_dist = (X - cx) * major_cos + (Y - cy) * major_sin
    # Distance along minor axis from center  
    minor_dist = (X - cx) * minor_cos + (Y - cy) * minor_sin
    
    # Determine sampling positions
    max_extent = int(min(cx, cy, nx - cx, ny - cy))
    positions = np.arange(-max_extent, max_extent + 1, 2)  # Sample every 2 pixels for speed
    
    velocities_result = []
    sigmas_result = []
    valid_positions = []
    
    # Store all individual points for detailed output
    all_points_data = {
        'x_pixel': [],
        'y_pixel': [],
        'major_axis_position': [],
        'radial_distance': [],
        'velocity': [],
        'sigma': [],
        'sn_zone': [],        # <-- ADD THIS
        'v_err': [],          # <-- ADD THIS
        'vd_err': [],         # <-- ADD THIS
        'band_center_position': [],
        'minor_axis_offset': []
    }
    
    print(f"INFO: Using {method} method for band statistics")
    print(f"INFO: Sampling {len(positions)} positions along major axis")
    
    for pos in positions:
        # Find all pixels within the band at this position
        in_band = (np.abs(major_dist - pos) <= 1.0) & (np.abs(minor_dist) <= band_width_pixels/2)
        
        if np.sum(in_band) > 0:
            # Get coordinates and values of pixels in this band
            band_x = X[in_band]
            band_y = Y[in_band] 
            band_vels = velocity_map[in_band]
            band_sigs = sigma_map[in_band] if sigma_map is not None else np.full(np.sum(in_band), np.nan)
            band_sn = sn_zone_map[in_band] if sn_zone_map is not None else np.full(np.sum(in_band), np.nan)

            band_major_pos = major_dist[in_band]
            band_minor_pos = minor_dist[in_band]
            
            # Remove invalid values
            valid_mask = np.isfinite(band_vels)
            valid_sn = band_sn[valid_mask]
            # Calculate errors based on S/N
            valid_v_err = calculate_velocity_error(valid_sn)
            valid_vd_err = calculate_dispersion_error(valid_sn)
            if np.sum(valid_mask) > 0:
                valid_x = band_x[valid_mask]
                valid_y = band_y[valid_mask] 
                valid_vels = band_vels[valid_mask]
                valid_sigs = band_sigs[valid_mask]
                valid_major_pos = band_major_pos[valid_mask]
                valid_minor_pos = band_minor_pos[valid_mask]
                
                # Calculate radial distances from center (basically use the Pythagoren theorem)
                valid_radial = np.sqrt((valid_x - cx)**2 + (valid_y - cy)**2)
                
                # Store all individual points
                all_points_data['x_pixel'].extend(valid_x.tolist())
                all_points_data['y_pixel'].extend(valid_y.tolist())
                all_points_data['major_axis_position'].extend(valid_major_pos.tolist())
                all_points_data['radial_distance'].extend(valid_radial.tolist())
                all_points_data['velocity'].extend(valid_vels.tolist())
                all_points_data['sigma'].extend(valid_sigs.tolist())
                all_points_data['band_center_position'].extend([pos] * len(valid_vels))
                all_points_data['minor_axis_offset'].extend(valid_minor_pos.tolist())
                all_points_data['sn_zone'].extend(valid_sn.tolist())
                all_points_data['v_err'].extend(valid_v_err.tolist())
                all_points_data['vd_err'].extend(valid_vd_err.tolist())
                
                # Calculate statistic based on method
                if method == 'mean':
                    vel_stat = np.mean(valid_vels)
                elif method == 'median':
                    vel_stat = np.median(valid_vels)
                elif method == 'weighted_mean':
                    valid_sig_mask = np.isfinite(valid_sigs) & (valid_sigs > 0)
                    if np.sum(valid_sig_mask) > 2:  # Need at least 3 points for weighted mean
                        weights = 1.0 / (valid_sigs[valid_sig_mask]**2 + 1e-10)
                        vel_stat = np.average(valid_vels[valid_sig_mask], weights=weights)
                    else:
                        vel_stat = np.mean(valid_vels)
                else:
                    vel_stat = np.mean(valid_vels)
                
                # Calculate sigma statistic
                valid_sig_values = valid_sigs[np.isfinite(valid_sigs)]
                sig_stat = np.mean(valid_sig_values) if len(valid_sig_values) > 0 else np.nan
                
                velocities_result.append(vel_stat)
                sigmas_result.append(sig_stat)
                valid_positions.append(pos)
    
    velocities_result = np.array(velocities_result)
    sigmas_result = np.array(sigmas_result)
    valid_positions = np.array(valid_positions)
    
    print(f"INFO: Advanced method extracted {len(valid_positions)} positions")
    print(f"INFO: Total individual spaxels used: {len(all_points_data['velocity'])}")
    
    return valid_positions, velocities_result, sigmas_result, all_points_data

def estimate_inclination_from_ba(ba):
    """
    Estimate inclination from axis ratio b/a.
    """
    try:
        ba = float(ba)
    except Exception:
        return None
    if not (0 < ba <= 1):
        return None
    return np.degrees(np.arccos(ba))

def save_all_data(filename_base, metadata, photometric_info, scales, spaxel_df, 
                  center, pa, inclination, rotation_curve_data, all_points_data=None):
    """
    Save all extracted data to multiple files, including individual spaxel points.
    """
    print(f"INFO: Saving all data with base filename: {filename_base}")
    
    # 1. Save all headers/metadata
    metadata_rows = []
    for hdu_name, hdu_meta in metadata.items():
        for key, info in hdu_meta.items():
            metadata_rows.append({
                'HDU': hdu_name,
                'Key': key,
                'Value': str(info['value']),
                'Comment': str(info['comment']),
                'Type': type(info['value']).__name__
            })
    
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_file = f"{filename_base}_all_headers.tsv"
    metadata_df.to_csv(metadata_file, sep='\t', index=False)
    print(f"INFO: Saved metadata to {metadata_file}")
    
    # 2. Save photometric/physical information
    photo_rows = []
    for key, value in {**photometric_info, **scales}.items():
        photo_rows.append({
            'Parameter': key,
            'Value': value,
            'Type': type(value).__name__
        })
    
    photo_df = pd.DataFrame(photo_rows)
    photo_file = f"{filename_base}_metadata.tsv"
    photo_df.to_csv(photo_file, sep='\t', index=False)
    print(f"INFO: Saved photometric info to {photo_file}")
    
    # 3. Save all spaxel data
    spaxel_file = f"{filename_base}_all_spaxels.csv"
    spaxel_df.to_csv(spaxel_file, index=False)
    print(f"INFO: Saved spaxel data to {spaxel_file}")
    
    # 4. Save rotation curve data
    if rotation_curve_data is not None and all_points_data is not None:
        pos, vel, sig = rotation_curve_data
        
        # Calculate average errors for each position
        v_err_avg = []
        vd_err_avg = []
        
        for p in pos:
            # Find all points near this position
            points_df = pd.DataFrame(all_points_data)
            nearby = points_df[np.abs(points_df['band_center_position'] - p) <= 1.0]
            
            if len(nearby) > 0 and 'v_err' in nearby.columns:
                v_err_avg.append(nearby['v_err'].mean())
                vd_err_avg.append(nearby['vd_err'].mean())
            else:
                v_err_avg.append(np.nan)
                vd_err_avg.append(np.nan)
        
        rotation_df = pd.DataFrame({
            'position_pixels': pos,
            'velocity_km_s': vel,
            'sigma_km_s': sig,
            'v_err_km_s': v_err_avg,          
            'vd_err_km_s': vd_err_avg,        
            'position_arcsec': pos * scales.get('pixel_scale_arcsec', np.nan),
            'position_kpc': pos * scales.get('kpc_per_pixel', np.nan)
        })
        
        # Save rotation curve
        rotation_file = f"{filename_base}_rotation_curve.csv"
        rotation_df.to_csv(rotation_file, index=False)
        print(f"INFO: Saved rotation curve to {rotation_file}")
        
        # Add kinematic info as comment in the file
        with open(rotation_file, 'r') as f:
            content = f.read()
        
        with open(rotation_file, 'w') as f:
            f.write(f"# Kinematic center: x={center[0] if center else 'NaN':.2f}, y={center[1] if center else 'NaN':.2f} pixels\n")
            f.write(f"# Position angle: {pa if pa is not None else 'NaN'}°\n")
            f.write(f"# Inclination: {inclination if inclination is not None else 'NaN'}°\n")
            f.write(f"# Pixel scale: {scales.get('pixel_scale_arcsec', 'NaN')} arcsec/pixel\n")
            f.write(f"# Physical scale: {scales.get('kpc_per_pixel', 'NaN')} kpc/pixel\n")
            f.write(f"# Band width: 3.0 arcsec\n")
            f.write(f"# Extraction method: Advanced band sampling\n")
            f.write(content)
    
    # 5. Save individual spaxel points used in rotation curve
    if all_points_data is not None:
        points_df = pd.DataFrame(all_points_data)
        
        # Add physical units
        if 'pixel_scale_arcsec' in scales:
            points_df['radial_distance_arcsec'] = points_df['radial_distance'] * scales['pixel_scale_arcsec']
            points_df['major_axis_position_arcsec'] = points_df['major_axis_position'] * scales['pixel_scale_arcsec']
        
        if 'kpc_per_pixel' in scales:
            points_df['radial_distance_kpc'] = points_df['radial_distance'] * scales['kpc_per_pixel']
            points_df['major_axis_position_kpc'] = points_df['major_axis_position'] * scales['kpc_per_pixel']
        
        # Sort by radial distance for easier analysis
        points_df = points_df.sort_values('radial_distance')
        
        points_file = f"{filename_base}_rotation_curve_points.csv"
        points_df.to_csv(points_file, index=False)
        print(f"INFO: Saved individual rotation curve points to {points_file}")
        
        # Add header comments
        with open(points_file, 'r') as f:
            content = f.read()
        
        with open(points_file, 'w') as f:
            f.write(f"# Individual spaxels used for rotation curve extraction\n")
            f.write(f"# Total points: {len(points_df)}\n")
            f.write(f"# Kinematic center: x={center[0] if center else 'NaN':.2f}, y={center[1] if center else 'NaN':.2f} pixels\n")
            f.write(f"# Position angle: {pa if pa is not None else 'NaN'}°\n")
            f.write(f"# Band width: 3.0 arcsec\n")
            f.write(f"# radial_distance: distance from kinematic center\n")
            f.write(f"# major_axis_position: signed distance along major axis (negative/positive sides)\n")
            f.write(f"# minor_axis_offset: perpendicular offset from major axis\n")
            f.write(content)

if __name__ == "__main__":
    print("=== ENHANCED PYCASSO VELOCITY EXTRACTOR STARTING ===")
    
    if len(sys.argv) < 2:
        print("Usage: python pycasso_enhanced_extractor.py <fits_file> [--show-map]")
        sys.exit(1)

    filename = sys.argv[1]
    show_map = '--show-map' in sys.argv
    
    # Create base filename for output files
    filename_base = Path(filename).stem
    
    print(f"INFO: Processing file: {filename}")
    print(f"INFO: Base filename for outputs: {filename_base}")
    print(f"INFO: Show map option: {show_map}")

    try:
        print("INFO: Opening FITS file...")
        with fits.open(filename) as hdul:
            print(f"INFO: FITS file opened successfully")
            
            # Get extension names/info
            extension_info = []
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'name') and hdu.name:
                    extension_info.append(f"{i}: {hdu.name}")
                else:
                    extension_info.append(f"{i}: HDU{i}")
            
            print(f"INFO: Available extensions: {extension_info}")
            
            # Read CALIFA photometric catalog
            print("\n=== READING CALIFA PHOTOMETRIC CATALOG ===")
            califa_catalog = read_califa_photometric_catalog()
            califa_phot_data = get_califa_photometric_data(filename, califa_catalog)
            
            # Extract ALL metadata first
            print("\n=== EXTRACTING ALL METADATA ===")
            metadata = extract_all_metadata(hdul)
            
            # Extract photometric and physical information (including CALIFA data)
            print("\n=== EXTRACTING PHOTOMETRIC INFORMATION ===")
            photometric_info, all_header_keys = extract_photometric_info(metadata, califa_phot_data)
            
            # Calculate physical scales
            print("\n=== CALCULATING PHYSICAL SCALES ===")
            scales = calculate_physical_scales(photometric_info)
            
            # Extract all spaxel data
            print("\n=== EXTRACTING ALL SPAXEL DATA ===")
            spaxel_df = extract_all_spaxel_data(hdul)
            
            # Check for required extensions for kinematic analysis
            extension_names = [hdu.name if hasattr(hdu, 'name') else f"HDU{i}" for i, hdu in enumerate(hdul)]
            if 'V_0' not in extension_names or 'V_D' not in extension_names:
                print("WARNING: FITS file does not contain V_0 and V_D extensions.")
                print(f"WARNING: Found extensions: {extension_names}")
                print("WARNING: Will skip kinematic analysis but save all other data.")
                velocity_map = None
                sigma_map = None
                sigma_star_map = None
                header = None
                sn_zone_map = None
            else:
                print("INFO: Loading velocity and sigma maps...")
                velocity_map = hdul['V_0'].data
                sigma_map = hdul['V_D'].data
                sigma_star_map = hdul['SIGMA_STAR'].data
                header = hdul['V_0'].header
                sn_zone_map = hdul['SN_ZONE'].data if 'SN_ZONE' in extension_names else None
                if sn_zone_map is not None:
                    print(f"INFO: SN_ZONE map loaded with shape: {sn_zone_map.shape}")
                else:
                    print("WARNING: SN_ZONE extension not found")
                
                print(f"INFO: Velocity map shape: {velocity_map.shape}")
                print(f"INFO: Sigma map shape: {sigma_map.shape}")
            
    except Exception as e:
        print(f"ERROR: Failed to open or read FITS file: {e}")
        sys.exit(1)

    # Kinematic analysis (if velocity data available)
    center = None
    pa = None
    inclination = None
    rotation_curve_data = None
    
    if velocity_map is not None:
        print("\n=== KINEMATIC ANALYSIS ===")
        print("Finding kinematic center (SIGMA PEAK METHOD PRIORITIZED)...")
        try:
            center = find_kinematic_center(velocity_map, sigma_star_map)
        except Exception as e:
            print(f"ERROR in find_kinematic_center: {e}")
            center = None
        

        print("\nEstimating inclination from axis ratio...")
        try:
            # Try to find axis ratio - prioritize CALIFA catalog data
            ba = None
            ba_source = "unknown"
            
            if 'califa_ba' in photometric_info:
                ba = photometric_info['califa_ba']
                ba_source = "CALIFA catalog"
            elif 'axis_ratio_ba' in photometric_info:
                ba = photometric_info['axis_ratio_ba']
                ba_source = "FITS header BA"
            elif header and 'BA' in header:
                ba = header['BA']
                ba_source = "primary header BA"
            
            if ba is not None:
                inclination = estimate_inclination_from_ba(ba)
                print(f"INFO: Found b/a ratio: {ba:.3f} (source: {ba_source})")
                if inclination is not None:
                    print(f"INFO: Estimated inclination: {inclination:.1f}°")
                else:
                    print("WARNING: Invalid b/a ratio for inclination calculation")
            else:
                print("WARNING: No b/a ratio found in CALIFA catalog, metadata, or header")
        except Exception as e:
            print(f"ERROR in inclination calculation: {e}")
            inclination = None
        
        # Using CALIFA position angle 
        califa_pa = None
        if 'califa_pa_align' in photometric_info:
            califa_pa = photometric_info['califa_pa_align']
            pa = 360 - califa_pa

        
        print(f"\n=== KINEMATIC PARAMETERS ===")
        if center:
            print(f"Kinematic center: x={center[0]:.1f} (col), y={center[1]:.1f} (row) [SIGMA PEAK METHOD]")
        else:
            print("Kinematic center: Not determined")
        if pa is not None:
            print(f"Position angle: {pa:.1f}°")
        else:
            print("Position angle: Not determined")
        if inclination is not None:
            print(f"Inclination: {inclination:.1f}°")
        else:
            print("Inclination: Not determined")
        print("="*50)

        # Extract rotation curve using advanced method
        print("\nINFO: Extracting major axis profile with advanced band sampling...")

        # Get pixel scale for band width calculation
        pixel_scale = scales.get('pixel_scale_arcsec', None)
        
        pos, vel, sig, all_points = extract_major_axis_profile_advanced(
            velocity_map, sigma_map, 
            center=center, pa=pa, inclination=inclination,
            band_width_arcsec=3.0,  # 3 arcsecond band width
            pixel_scale_arcsec=pixel_scale,
            method='weighted_mean',  # Can be 'mean', 'median', or 'weighted_mean'
            sn_zone_map=sn_zone_map 
        )
        rotation_curve_data = (pos, vel, sig)
        
        print(f"INFO: Extracted {len(pos)} points along major axis using advanced band sampling")
        print(f"INFO: Individual spaxels used: {len(all_points['velocity'])}")
              
    # Update spaxel data with kinematic center-based radii
    print("\n=== UPDATING SPAXEL DATA WITH KINEMATIC RADII ===")
    if not spaxel_df.empty:
        # Keep the original geometric radius for comparison
        spaxel_df.rename(columns={'radius_pixels': 'radius_pixels_geometric'}, inplace=True)
        
        # Calculate kinematic radius if center was found
        if center is not None:
            cx_kin, cy_kin = center
            spaxel_df['radius_pixels_kinematic'] = np.sqrt(
                (spaxel_df['x_pixel'] - cx_kin)**2 + 
                (spaxel_df['y_pixel'] - cy_kin)**2
            )
            print(f"INFO: Added kinematic radius based on center: x={cx_kin:.1f}, y={cy_kin:.1f}")
            
            # Use kinematic radius as the primary radius
            spaxel_df['radius_pixels'] = spaxel_df['radius_pixels_kinematic']
        else:
            print("WARNING: No kinematic center found, using geometric radius")
            spaxel_df['radius_pixels'] = spaxel_df['radius_pixels_geometric']
        
        # Add physical coordinates using the corrected radii
        if 'pixel_scale_arcsec' in scales:
            spaxel_df['radius_arcsec'] = spaxel_df['radius_pixels'] * scales['pixel_scale_arcsec']
            print(f"INFO: Added radius in arcsec (scale: {scales['pixel_scale_arcsec']:.3f} arcsec/pixel)")
        
        if 'kpc_per_pixel' in scales:
            spaxel_df['radius_kpc'] = spaxel_df['radius_pixels'] * scales['kpc_per_pixel']
            print(f"INFO: Added radius in kpc (scale: {scales['kpc_per_pixel']:.4f} kpc/pixel)")
        
        # Add comparison info
        if center is not None:
            ny, nx = spaxel_df['y_pixel'].max() + 1, spaxel_df['x_pixel'].max() + 1
            geom_center_x, geom_center_y = nx / 2.0, ny / 2.0
            center_offset = np.sqrt((center[0] - geom_center_x)**2 + (center[1] - geom_center_y)**2)
            print(f"INFO: Kinematic center offset from geometric: {center_offset:.1f} pixels")
            
            # Show some statistics
            max_geom_r = spaxel_df['radius_pixels_geometric'].max()
            max_kin_r = spaxel_df['radius_pixels_kinematic'].max()
            print(f"INFO: Maximum geometric radius: {max_geom_r:.1f} pixels")
            print(f"INFO: Maximum kinematic radius: {max_kin_r:.1f} pixels")
    
    else:
        print("WARNING: No spaxel data available for radius updates")
    
    # Save all data to files
    print("\n=== SAVING ALL DATA ===")
    try:
        save_all_data(filename_base, metadata, photometric_info, scales, spaxel_df, 
                      center, pa, inclination, rotation_curve_data, 
                      all_points_data=all_points if velocity_map is not None else None)
    except Exception as e:
        print(f"ERROR saving data: {e}")


    
    print("\n=== COMPREHENSIVE ANALYSIS COMPLETE ===")
    print(f"Output files saved with base name: {filename_base}")
    print("Files created:")
    print(f"  - {filename_base}_all_headers.tsv (all FITS headers)")
    print(f"  - {filename_base}_metadata.tsv (photometric & physical parameters)")
    print(f"  - {filename_base}_all_spaxels.csv (complete spaxel data)")
    if rotation_curve_data is not None:
        print(f"  - {filename_base}_rotation_curve.csv (averaged rotation curve)")
        print(f"  - {filename_base}_rotation_curve_points.csv (individual spaxel points)")
    print("="*60)