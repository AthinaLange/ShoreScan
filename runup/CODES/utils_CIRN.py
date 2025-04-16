"""
utils_CIRN.py

This module provides functions for rectifing imagery - based on CIRN QCIT toolbox

"""
import json
import math
import os
from tkinter import Tk, filedialog, messagebox

import cv2
import numpy as np
import scipy.ndimage
from scipy import interpolate
import utm


def CIRNangles2R(azimuth, tilt, roll):
    """
    Computes a 3x3 rotation matrix R to transform world coordinates to camera coordinates using a ZXZ rotation sequence.

    :param azimuth: (float) Horizontal rotation (radians), positive CW from world Z-axis.
    :param tilt: (float) Up/down tilt (radians), 0 is nadir, +90 is horizon.
    :param roll: (float) Side-to-side tilt (radians), 0 is a horizontal flat camera.
    :return: (np.ndarray) 3x3 rotation matrix.
    """

    ## Section 1: Define Rotation Matrix R
    R = np.zeros((3, 3))

    R[0, 0] = -np.cos(azimuth) * np.cos(roll) - np.sin(azimuth) * np.cos(tilt) * np.sin(roll)
    R[0, 1] = np.cos(roll) * np.sin(azimuth) - np.sin(roll) * np.cos(tilt) * np.cos(azimuth)
    R[0, 2] = -np.sin(roll) * np.sin(tilt)

    R[1, 0] = -np.sin(roll) * np.cos(azimuth) + np.cos(roll) * np.cos(tilt) * np.sin(azimuth)
    R[1, 1] = np.sin(roll) * np.sin(azimuth) + np.cos(roll) * np.cos(tilt) * np.cos(azimuth)
    R[1, 2] = np.cos(roll) * np.sin(tilt)

    R[2, 0] = np.sin(tilt) * np.sin(azimuth)
    R[2, 1] = np.sin(tilt) * np.cos(azimuth)
    R[2, 2] = -np.cos(tilt)

    return R

def undistortUV(Ud, Vd, intrinsics):
    """
    Undistorts distorted UV coordinates using distortion models from the Caltech lens distortion manuals.

    :param Ud: (np.ndarray) Px1 array of distorted U coordinates.
    :param Vd: (np.ndarray) Px1 array of distorted V coordinates.
    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :return: tuple(np.ndarray, np.ndarray) Px1 arrays of undistorted U and V coordinates.
    """
    
    ## Section 1: Define Coefficients
    fx, fy, c0U, c0V = intrinsics['fx'], intrinsics['fy'], intrinsics['coU'], intrinsics['coV']
    d1, d2, d3, t1, t2 = intrinsics['d1'], intrinsics['d2'], intrinsics['d3'], intrinsics['t1'], intrinsics['t2']
    
    ## Section 2: Provide first guess for dx, dy, and fr using distorted x,y
    # Calculate Distorted camera coordinates x, y, and r
    xd = (Ud - c0U) / fx
    yd = (Vd - c0V) / fy
    rd = np.sqrt(xd**2 + yd**2)
    r2d = rd**2

    # Calculate First Guess for Aggregate Coefficients
    fr1 = 1 + d1 * r2d + d2 * r2d**2 + d3 * r2d**3
    dx1 = 2 * t1 * xd * yd + t2 * (r2d + 2 * xd**2)
    dy1 = t1 * (r2d + 2 * yd**2) + 2 * t2 * xd * yd

    ## Section 3: Calculate Undistorted X and Y using first guess
    x = (xd - dx1) / fr1
    y = (yd - dy1) / fr1

    ## Section 4: Iterate until the difference for all values is < 0.001%
    while True:
        # Calculate New Coefficients
        rn = np.sqrt(x**2 + y**2)
        r2n = rn**2
        frn = 1 + d1 * r2n + d2 * r2n**2 + d3 * r2n**3
        dxn = 2 * t1 * x * y + t2 * (r2n + 2 * x**2)
        dyn = t1 * (r2n + 2 * y**2) + 2 * t2 * x * y

        # Determine Percent change from previous fr, dx, and dy values
        chk1 = np.abs(100 * (fr1 - frn) / fr1)
        chk2 = np.abs(100 * (dx1 - dxn) / dx1)
        chk3 = np.abs(100 * (dy1 - dyn) / dy1)

        # Check if all percent changes are less than 0.001%
        if np.all(chk1 < 0.001) and np.all(chk2 < 0.001) and np.all(chk3 < 0.001):
            break

        # Update x, y for the next iteration
        x = (xd - dxn) / frn
        y = (yd - dyn) / frn

        # Update coefficients for the next iteration
        fr1, dx1, dy1 = frn, dxn, dyn

    ## Section 5: Convert x and y to U, V
    U = x * fx + c0U
    V = y * fy + c0V

    return U, V

def distortUV(U, V, intrinsics):
    """
    Distorts undistorted UV coordinates using the distortion model.
    
    :param U (ndarray): Px1 array of undistorted U coordinates.
    :param V (ndarray): Px1 array of undistorted V coordinates.
    :param intrinsics (dict): Dictionary containing the intrinsic camera parameters, including focal length, 
        principal point, and distortion coefficients.

    :return: tuple
        - Ud (ndarray): Px1 array of distorted U coordinates.
        - Vd (ndarray): Px1 array of distorted V coordinates.
        - flag (ndarray): Px1 array indicating if the UVd coordinate is valid (1) or not (0).
    """
    
    fx, fy, c0U, c0V = intrinsics['fx'], intrinsics['fy'], intrinsics['coU'], intrinsics['coV']
    d1, d2, d3, t1, t2 = intrinsics['d1'], intrinsics['d2'], intrinsics['d3'], intrinsics['t1'], intrinsics['t2']
    
    x = (U - c0U) / fx
    y = (V - c0V) / fy
    r2 = x**2 + y**2
    fr = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3
    dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
    dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y
    
    xd = x * fr + dx
    yd = y * fr + dy
    
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V
    
    flag = (r2 < 1).astype(int)  # Validity check based on a threshold
    
    return Ud, Vd, flag

def intrinsicsExtrinsics2P(intrinsics, extrinsics):
    """
    Computes a camera projection matrix from camera intrinsics and extrinsics.

    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary containing [x, y, z, azimuth, tilt, roll] of the camera (azimuth, tilt, roll should be in degrees).
    :return: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray) Transformation matrices P, K, R, and IC.
    
    Notes:
        - P: 4x4 transformation matrix to convert XYZ coordinates to distorted UV coordinates
        - K: 3x3 intrinsic matrix to convert XYZc coordinates to distorted UV coordinates
        - R: 3x3 rotation matrix to rotate world XYZ to camera coordinates XYZc
        - IC: 4x3 translation matrix to convert world XYZ to camera coordinates XYZc
    """

    ## Section 1: Format IO into K matrix
    fx, fy, c0U, c0V = intrinsics['fx'], intrinsics['fy'], intrinsics['coU'], intrinsics['coV']
    
    K = np.array([
        [-fx,  0,  c0U],
        [  0, -fy,  c0V],
        [  0,   0,    1]
    ])

    ## Section 2: Compute Rotation Matrix R using CIRN defined angles
    x, y, z = extrinsics['x'], extrinsics['y'], extrinsics['z']
    azimuth, tilt, roll = np.radians([extrinsics['azimuth'], extrinsics['tilt'], extrinsics['roll']])
    R = CIRNangles2R(azimuth, tilt, roll)  # Ensure CIRNangles2R is defined

    ## Section 3: Compute Translation Matrix IC
    IC = np.hstack((np.eye(3), np.array([[-x], [-y], [-z]])))  # 3x4 matrix

    ## Section 4: Compute Camera Projection Matrix P
    P = K @ R @ IC  # Matrix multiplication
    P /= P[2, 3]  # Normalize for homogeneous coordinates

    return P, K, R, IC

def uv_to_xyz(intrinsics, extrinsics, Ud, Vd, known_dim, known_val):
    """
    Converts image coordinates (U, V) to world coordinates (X, Y, Z) using camera intrinsics, extrinsics, and the Direct Linear Transformation (DLT) equations.

    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary of extrinsic parameters [x, y, z, azimuth, tilt, roll].
    :param Ud: (np.ndarray) Array of distorted U image coordinates.
    :param Vd: (np.ndarray) Array of distorted V image coordinates.
    :param known_dim: (str) The known world coordinate dimension ('x', 'y', or 'z').
    :param known_val: (np.ndarray) The known value of the world coordinate.
    :return: (np.ndarray) Nx3 NumPy array of computed world coordinates [X, Y, Z].
    """

    # Step 1: Convert UV to undistorted image coordinates
    U=Ud
    V=Vd

    # Step 2: Compute camera projection matrix P
    P, _, _, _ = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    known_val = np.array(known_val)
    if known_val.ndim == 0:  # If it's a single integer (scalar)
        known_val = np.full(U.shape, known_val)
    
    # Extract DLT coefficients from P matrix
    A, B, C, D = P[0, :4]
    H, J, K, L = P[1, :4]
    E, F, G = P[2, :3]  # Only 3 elements since the 4th is always 1 in homogeneous coordinates

    # Compute intermediate variables
    M, N, O, P_ = (E * U - A), (F * U - B), (G * U - C), (D - U)
    Q, R, S, T = (E * V - H), (F * V - J), (G * V - K), (L - V)

    # Solve for world coordinates based on the known dimension
    if known_dim.lower() == 'x':
        X = known_val
        Y = ((O * Q - S * M) * X + (S * P_ - O * T)) / (S * N - O * R)
        Z = ((N * Q - R * M) * X + (R * P_ - N * T)) / (R * O - N * S)
    elif known_dim.lower() == 'y':
        Y = known_val
        X = ((O * R - S * N) * Y + (S * P_ - O * T)) / (S * M - O * Q)
        Z = ((M * R - Q * N) * Y + (Q * P_ - M * T)) / (Q * O - M * S)
    elif known_dim.lower() == 'z':
        Z = known_val
        X = ((N * S - R * O) * Z + (R * P_ - N * T)) / (R * M - N * Q)
        Y = ((M * S - Q * O) * Z + (Q * P_ - M * T)) / (Q * N - M * R)
    else:
        raise ValueError("known_dim must be 'x', 'y', or 'z'")

    # Ensure consistent output format
    max_size = max(map(np.size,[X,Y,Z]))
    X = np.full(max_size, X) if np.size(X) == 1 else X
    Y = np.full(max_size, Y) if np.size(Y) == 1 else Y
    Z = np.full(max_size, Z) if np.size(Z) == 1 else Z
   
    xyz = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    return xyz

def dist_uv_to_xyz(intrinsics, extrinsics, Ud, Vd, known_dim, known_val, timeout = 60):
    """
    Converts image coordinates (U, V) to world coordinates (X, Y, Z) using camera intrinsics, extrinsics, and the Direct Linear Transformation (DLT) equations.

    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary of extrinsic parameters [x, y, z, azimuth, tilt, roll].
    :param Ud: (np.ndarray) Array of distorted U image coordinates.
    :param Vd: (np.ndarray) Array of distorted V image coordinates.
    :param known_dim: (str) The known world coordinate dimension ('x', 'y', or 'z').
    :param known_val: (np.ndarray) The known value of the world coordinate.
    :return: (np.ndarray) Nx3 NumPy array of computed world coordinates [X, Y, Z].
    """

    # Step 1: Convert UV to undistorted image coordinates
    U, V = undistortUV(Ud, Vd, intrinsics)

    # Step 2: Compute camera projection matrix P
    P, _, _, _ = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    # Extract DLT coefficients from P matrix
    A, B, C, D = P[0, :4]
    H, J, K, L = P[1, :4]
    E, F, G = P[2, :3]  # Only 3 elements since the 4th is always 1 in homogeneous coordinates

    # Compute intermediate variables
    M, N, O, P_ = (E * U - A), (F * U - B), (G * U - C), (D - U)
    Q, R, S, T = (E * V - H), (F * V - J), (G * V - K), (L - V)

    # Solve for world coordinates based on the known dimension
    if known_dim == 'x':
        X = known_val
        Y = ((O * Q - S * M) * X + (S * P_ - O * T)) / (S * N - O * R)
        Z = ((N * Q - R * M) * X + (R * P_ - N * T)) / (R * O - N * S)
    elif known_dim == 'y':
        Y = known_val
        X = ((O * R - S * N) * Y + (S * P_ - O * T)) / (S * M - O * Q)
        Z = ((M * R - Q * N) * Y + (Q * P_ - M * T)) / (Q * O - M * S)
    elif known_dim == 'z':
        Z = known_val
        X = ((N * S - R * O) * Z + (R * P_ - N * T)) / (R * M - N * Q)
        Y = ((M * S - Q * O) * Z + (Q * P_ - M * T)) / (Q * N - M * R)
    else:
        raise ValueError("known_dim must be 'x', 'y', or 'z'")

    max_size = max(map(np.size,[X,Y,Z]))
    X = np.full(max_size, X) if np.size(X) == 1 else X
    Y = np.full(max_size, Y) if np.size(Y) == 1 else Y
    Z = np.full(max_size, Z) if np.size(Z) == 1 else Z
   
    # Ensure consistent output format
    xyz = np.column_stack((X.ravel(),Y.ravel(), Z.ravel()))

    return xyz

def xyz_to_dist_uv(intrinsics, extrinsics, xyz):
    """
    Computes the distorted UV coordinates (UVd) for a set of world xyz points given camera EO and IO.
    
    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary of extrinsic parameters [x, y, z, azimuth, tilt, roll].
    :param xyz: (np.ndarray) Nx3 NumPy array of computed world coordinates [X, Y, Z]
    :returns UVd: (np.ndarray) 2Px1 array of distorted UV coordinates.
    """

    # Compute P matrix containing both intrinsics and extrinsics information
    P, K, R, IC = intrinsicsExtrinsics2P(intrinsics, extrinsics)
    
    # Compute undistorted UV coordinates
    xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))  # Convert to homogeneous coordinates
    UV = P @ xyz_homogeneous.T
    UV /= UV[2, :]  # Normalize to make homogeneous
    
    # Distort the undistorted UV coordinates
    Ud, Vd, flag = distortUV(UV[0, :], UV[1, :], intrinsics)
    flag = np.ones(Ud.shape[0])
    # Compute camera coordinates and check for negative Z values
    xyzC = R @ IC @ xyz_homogeneous.T
    flag[xyzC[2, :] <= 0] = 0
    
    # Stack Ud and Vd to form a single matrix
    UVd = np.vstack((Ud, Vd))
    
    return UVd, flag

def get_site_settings(file_path = None):
    """
    Load site settings from a JSON file and convert date strings back to datetime objects.
    The JSON file should be named `site_settings.json` and structured as follows:
    
    ```json
    {
        "SITE_ID": {  // Each site (e.g., "CACO03", "SITEB") is a key containing site-specific information
            "siteName": "Full site name",  // Descriptive name of the site
            "shortName": "Short identifier",  // Abbreviated site name
            "siteInfo": {  // Metadata related to the site
                "siteLocation": "Geographical location of the site",
                "dataOrigin": "Organization responsible for the data",
                "camMake": "Camera manufacturer",
                "camModel": "Camera model",
                "camLens": "Lens specifications",
                "timezone": "Local timezone",
                "utmZone": "UTM coordinate zone",
                "verticalDatum": "Vertical reference system",
                "verticalDatum_description": "Description of the vertical datum",
                "references": "Citation or source reference for the data",
                "contributors": "Names of individuals who contributed to data collection",
                "metadata_link": "URL to metadata and dataset information"
            },
            "sampling": {  // Sampling configuration for data collection
                "sample_frequency": Number,  // Sampling frequency (e.g., 2, 5)
                "collection_unit": "Unit of frequency (Hz, seconds, etc.)",
                "sample_period_length": Number,  // Duration of each sampling period
                "sample_period_unit": "Unit for sample period (s, min, etc.)",
                "freqLimits": [  // Frequency range limits (if applicable)
                    Upper SS limit,
                    SS/IG transition limit,
                    Lower IG limit
                ]
            }
        }
    }
    Load site settings from a JSON file.

    :param file_path: (str, optional) Path to `site_settings.json`. If None, prompts the user to select a directory.
    :return: (dict) Parsed site settings.
    :raises FileNotFoundError: If the file is not found.
    :raises ValueError: If the JSON file is empty or malformed.
=
    """
    
    # If no file path is provided, prompt the user to select a directory
    if file_path is None:
        Tk().withdraw()  # Hide the root Tkinter window
        directory = filedialog.askdirectory(title="Select the directory containing site_settings.json")
        
        if not directory:  # If the user cancels, exit
            raise FileNotFoundError("No directory selected.")
        
        file_path = os.path.join(directory, "site_settings.json")  # Look for the JSON file in the selected directory

    # Ensure the file exists
    if not os.path.exists(file_path):
        try:
            Tk().withdraw()  # Hide the root Tkinter window
            directory = filedialog.askdirectory(title="Select the directory containing site_settings.json")
            file_path = os.path.join(directory, "site_settings.json")
        except:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load JSON file
    try:
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing JSON file: {file_path}")

    return loaded_data  # Return the entire structure

def get_pixels(output_grid, UV_coords, I):
    """
    Computes pixel coordinates (UV) from world coordinates (XYZ) and extracts pixel values from an image.

    :param products (dict): Dictionary containing transect data.
    :param extrinsics (dict): Camera extrinsic parameters.
    :param intrinsics (dict): Camera intrinsic parameters.
    :param I (numpy.ndarray): Image matrix representing the image data to be processed.

    :return: dict
        - A dictionary containing processed pixel data for each transect, including:
            - "Ir" (ndarray): Rectified image values.
            - "localX" (ndarray), "localY" (ndarray): Local coordinates.
            - "Z" (ndarray): Elevation values.
            - "Eastings" (ndarray), "Northings" (ndarray): UTM coordinates.
    """
    
    for key, data in output_grid.items():
        Xout = data['localX']
        Ud, Vd = UV_coords[key]  # Use precomputed UV coordinates
        valid_mask = (
            ~np.isnan(Ud) & ~np.isnan(Vd) &
            (Ud >= 0) & (Ud < I.shape[1]) &
            (Vd >= 0) & (Vd < I.shape[0])
        )

        # Initialize rectified image array
        Ir = np.zeros((Xout.shape[0], Xout.shape[1], 3), dtype=np.uint8)
        
        # Extract pixel values
        for kk in range(Xout.shape[0]):
            for j in range(Xout.shape[1]):
                if valid_mask[kk, j]:
                    ir_value = I[Vd[kk, j].astype(int), Ud[kk, j].astype(int), :]
                
                    # Check if the extracted value is valid
                    if np.all(np.isfinite(ir_value)):  # Check if all values are finite (no NaNs or Infs)
                        Ir[kk, j, :] = ir_value
                    else:
                        Ir[kk, j, :] = [0, 0, 0]  # Set to default valid value (e.g., black pixel)

        
        Ir,_ = filter_regions(Ir)
        output_grid[key]["Ir"] = Ir.tolist()
    
    return output_grid

def get_xy_coords(products):
    """
    Computes XY coordinates and other associated data based on the provided products dictionary.

    :param products (dict): Dictionary containing the following fields:
        - 'lat' (list or ndarray): Latitude values.
        - 'lon' (list or ndarray): Longitude values.
        - 'angle' (float): Rotation angle in degrees.
        - 'tide' (float or ndarray): Tide values.
        - 'xlim' (list or tuple): Limits for the X coordinates (min, max).
        - 'ylim' (list or tuple): Limits for the Y coordinates (min, max).
        - 'type' (str): The type of transect data, such as 'Grid', 'xTransect', or 'yTransect'.
        - 'x' (ndarray): X coordinates for transect data.
        - 'y' (ndarray): Y coordinates for transect data.
        - Optional: 'east' (ndarray), 'north' (ndarray), 'z' (list or ndarray), 'lim_flag' (int).
        
    :return: dict
        - A dictionary containing the processed XY and associated data:
            - 'xyz' (ndarray): Combined XY and Z coordinates.
            - 'localX' (ndarray): Local X coordinates.
            - 'localY' (ndarray): Local Y coordinates.
            - 'Elevation' (ndarray): Elevation values.
            - 'Eastings' (ndarray): UTM Easting coordinates.
            - 'Northings' (ndarray): UTM Northing coordinates.
            - 'local_grid_origin' (ndarray): Origin of the local grid (Easting, Northing).
            - 'local_grid_angle' (float): Rotation angle for the local grid.
    """
    assert isinstance(products, dict), "Error: Products must be a dictionary."
    required_fields = ['lat', 'lon', 'angle', 'tide', 'xlim', 'ylim', 'type', 'x', 'y']
    for field in required_fields:
        assert field in products, f"Error: Products must have {field} field."

    if {'east', 'north'}.issubset(products):
        easting = products["east"]
        northing = products["north"]
    else:
        easting, northing, _,_ = utm.from_latlon(products['lat'], products['lon'])


    if {'lim_flag'}.issubset(products): # are limits defined in UTM or local coordinates
        flag = products['lim_flag']
    else:
        if math.floor(math.log10(abs(np.max(products['xlim'])))) == math.floor(math.log10(abs(easting))):
            flag = 1 # geographical definition
        else:
            flag = 0 # local definition

    if products.get('z') is None:
        iz = products['tide']
    elif isinstance(products['z'], list):
        iz = [z + products['tide'] for z in products['z']]  # Element-wise addition
    else:
        iz = products['z'] + products['tide']

    results = []
    if 'Grid' in products['type']:
        X, Y = np.meshgrid(
                np.arange(products['xlim'][0], products['xlim'][1], products['dx']),
                np.arange(products['ylim'][0], products['xlim'][1], products['dy'])
            )
        results.append({'X': X, 'Y': Y})
    elif 'xTransect' in products['type']:
        for y_val in products['y']:
            X = np.arange(products['xlim'][0], products['xlim'][1], products['dx'])[:, None]
            Y = np.full_like(X, y_val)
            results.append({'X': X, 'Y': Y, 'transect': y_val})
    elif 'yTransect' in products['type']:
        for x_val in products['x']:
            Y = np.arange(products['ylim'][0], products['ylim'][1], products['dy'])[:, None]
            X = np.full_like(Y, x_val)
            results.append({'X': X, 'Y': Y, 'transect': x_val})
    else:
        raise ValueError("Error: Unrecognized type in products['type']")

    output = {}
    for i, result in enumerate(results):
        X, Y = result['X'], result['Y']
        transect = result.get('transect', i)
        
        if flag == 0:
            Eastings, Northings = local_transform_points(easting, northing, np.radians(products["angle"]), flag, X, Y)
            localX = X
            localY = Y
        else:
            localX, localY = local_transform_points(easting, northing, np.radians(products["angle"]), flag, X, Y)
            Eastings = X
            Northings = Y
        
        if isinstance(iz, np.ndarray) and iz.size != 1 and iz.shape != X.shape:
            Z = np.full(localX.shape, iz.mean())
        else:
            Z = np.full(localX.shape, iz)
        
        xyz = np.column_stack((Eastings.ravel(), Northings.ravel(), Z.ravel()))
        output[f'transect_{transect}'] = {
            'xyz': xyz,
            'localX': localX,
            'localY': localY,
            'Elevation': Z,
            'Eastings': Eastings,
            'Northings': Northings,
            'local_grid_origin': np.array([easting, northing]),
            'local_grid_angle': products["angle"]
        }
    
    return output

def get_uv_coords(output_grid, intrinsics, extrinsics):
    """
    Computes UV coordinates from XYZ coordinates and returns them for reuse.

    :param output_grid (dict): Dictionary containing processed XYZ data for each transect.
        - Each entry in the dictionary has the following fields:
            - 'xyz' (ndarray): Combined XYZ coordinates.
            - 'localX' (ndarray): Local X coordinates.
            - 'localY' (ndarray): Local Y coordinates.
    :param intrinsics (dict): Camera intrinsic parameters containing:
        - 'NU' (int): Number of columns in the image.
        - 'NV' (int): Number of rows in the image.
    :param extrinsics (dict): Camera extrinsic parameters.

    :return: dict
        - A dictionary containing UV coordinates for each transect:
            - Each entry corresponds to a key from `output_grid` and contains:
                - Ud (ndarray): Distorted U coordinates.
                - Vd (ndarray): Distorted V coordinates.
    """
    UV_coords = {}
    
    for key, data in output_grid.items():
        xyz = data['xyz']
        Xout = data['localX']
        
        UVd, flag = xyz_to_dist_uv(intrinsics, extrinsics, xyz)
        
        if UVd is None:
            continue
        
        s = Xout.shape
        Ud, Vd = UVd[0, :].reshape(s), UVd[1, :].reshape(s)
        
        # Ensure Ud and Vd are within valid image bounds
        valid_mask = (
            ~np.isnan(Ud) & ~np.isnan(Vd) &
            (Ud >= 0) & (Ud < intrinsics['NU']) &
            (Vd >= 0) & (Vd < intrinsics['NV'])
        )
        
        Ud[~valid_mask], Vd[~valid_mask] = np.nan, np.nan
        mask = ((Vd < 0) & (Ud < 0) )
        Ud[mask] = np.nan
        Vd[mask] = np.nan

        UV_coords[key] = (Ud, Vd)
    
    return UV_coords

def get_interp_dem(dem):
    """
    Generates an interpolation function from a given Digital Elevation Model (DEM).
    
    :param dem: (xarray.DataArray) The DEM containing x, y, and elevation values.
    
    :returns (function) A RegularGridInterpolator function for elevation lookup.
    """
    x = dem['x'].values
    y = np.flipud(dem['y'].values)
    z = np.flipud(np.squeeze(dem.values))

    # Create interpolation function
    interp_func = interpolate.RegularGridInterpolator(
        (x,y), z.T, bounds_error=False, fill_value=np.nan
    )
    return interp_func

def get_elevation_from_dem(interp_func, transect_data, camera_origin, initial_step_size = -1, max_steps = 250, max_error = 0.1, min_depth = -2):
    """
    Computes terrain intersection points using ray marching from a given camera origin.
    
    :param interp_func: (function) Interpolation function for terrain height lookup.
    :param transect_data: (dict) Dictionary containing xyz coordinate data.
    :param camera_origin: (np.array) The camera's origin point in 3D space.
    :param initial_step_size: (float) Initial step size for ray marching.
    :param max_steps: (int) Maximum number of steps before termination.
    :param min_error: (float) Minimum acceptable error for terrain intersection.
    :param min_depth: (float) Minimum depth allowed before terminating search.
    
    :returns: (tuple) Updated xyz points and a list of ray paths.
    """
    xyz_points = np.full_like(transect_data['xyz'], np.nan)
    xyz_ray=[]

    for i, point in enumerate(transect_data['xyz']):
        direction = point - camera_origin  # Vector from camera to point
        direction = direction / direction[-1]  # Normalize to unit scale
        step_size = -np.abs(initial_step_size)  # Ensure movement is downward

        # Generate ray points for visualization
        ray_points = np.array([np.outer(np.linspace(0, -camera_origin[2], 100), direction) + camera_origin])
        xyz_ray.append(ray_points.squeeze())
        
        # Perform ray marching to find terrain intersection
        pos = camera_origin.astype('float64').copy()
        for _ in range(max_steps):
            prev_pos = pos.copy()
            pos += step_size * direction.astype('float64')
            try:
                terrain_height = interp_func((pos[0], pos[1]))
                error = np.abs(pos[2] - terrain_height)
                if pos[2] <= terrain_height:
                    if error < max_error:  # Terrain hit condition
                        break
                    else:
                        step_size *= 0.5  # Reduce step size for accuracy
                        pos = prev_pos  # Revert to last position
                elif pos[2] <= min_depth:
                    pos = np.array([np.nan, np.nan, np.nan])  # Mark as invalid point
                    break
            except ValueError:
                break  # Stop if out of bounds
        xyz_points[i, :] = pos

    return xyz_points, xyz_ray
        
def get_elevations(dem, extrinsics, coords, max_error = 0.1):
    """
    Computes elevations for a set of coordinates using ray-marching and a DEM.
    
    Parameters:
        dem (xarray.DataArray): The Digital Elevation Model.
        extrinsics (dict): Dictionary containing camera extrinsic parameters (x, y, z).
        coords (dict): Dictionary containing xyz coordinates for elevation retrieval.
    
    Returns:
        tuple: Updated coordinate dictionary with elevation values and ray paths.
    """
    interp_func = get_interp_dem(dem)
    camera_origin = np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])
    
    if isinstance(coords, dict) and all(isinstance(v,dict) for v in coords.values()):
        for transect_key in coords:
            transect_data = coords[transect_key]
            if np.max(np.shape(transect_data['xyz'])) > 10000:
                root = Tk()
                root.withdraw()  
                response = messagebox.askyesno("Confirmation", "Your file is very big. Are you sure you want to continue?")
                root.destroy()
                if not response:
                    xyz_ray = []
                    continue
            xyz_points, xyz_ray = get_elevation_from_dem(interp_func, transect_data, camera_origin, min_depth = np.nanmin(np.squeeze(dem.values)), max_error=max_error)
            
            # Replace non-NaN values in coords['Eastings'] with values from xyz_points[:, 0]
            mask = ~np.isnan(xyz_points)  # Create a boolean mask for non-NaN values
            # Apply the mask to each column of the xyz array
            coords[transect_key]['xyz'][mask] = xyz_points[mask]
            coords[transect_key]['Eastings'] = coords[transect_key]['xyz'][:, 0]
            coords[transect_key]['Northings'] = coords[transect_key]['xyz'][:, 1]
            coords[transect_key]['Elevation'] = coords[transect_key]['xyz'][:, 2]
            coords[transect_key]['localX'], coords[transect_key]['localY'] = local_transform_points(coords[transect_key]['local_grid_origin'][0], coords[transect_key]['local_grid_origin'][1], np.radians(coords[transect_key]['local_grid_angle']), 1, coords[transect_key]['xyz'][:, 0], coords[transect_key]['xyz'][:, 1])
            
    else:
        if np.max(np.shape(coords['xyz'])) > 10000:
            root = Tk()
            root.withdraw()  
            response = messagebox.askyesno("Confirmation", "Your file is very big. Are you sure you want to continue?")
            root.destroy()
            if not response:
                xyz_ray = []
                raise Exception("Process stopped by user")
        xyz_points, xyz_ray = get_elevation_from_dem(interp_func, coords, camera_origin, min_depth = np.nanmin(np.squeeze(dem.values)), max_error=max_error)
        # Replace non-NaN values in coords['Eastings'] with values from xyz_points[:, 0]
        mask = ~np.isnan(xyz_points)  # Create a boolean mask for non-NaN values
        # Apply the mask to each column of the xyz array
        coords['xyz'][mask] = xyz_points[mask]
        coords['Eastings'] = coords['xyz'][:, 0]
        coords['Northings'] = coords['xyz'][:, 1]
        coords['Elevation'] = coords['xyz'][:, 2]
        coords['localX'], coords['localY'] = local_transform_points(coords['local_grid_origin'][0], coords['local_grid_origin'][1], np.radians(coords['local_grid_angle']), 1, coords['xyz'][:, 0], coords['xyz'][:, 1])
       
    return coords, xyz_ray

def camera_seam_blend(IrIndv):
    """
    Blends rectifications from different cameras into a single rectified image 
    using a weighted average, where pixels closer to the center of each camera's 
    rectification contribute more than those near the seams.
    
    :param IrIndv: A (N, M, C, K) NumPy array where:
                   - N, M are grid dimensions of the rectified image.
                   - C is the number of color channels (1 for grayscale, 3 for RGB).
                   - K is the number of cameras.
    :return: Ir (N, M, C) blended rectified image as a uint8 NumPy array.
    """

    # Get dimensions
    m, n, c, camnum = IrIndv.shape

    # Convert to float for calculations
    IrIndv = IrIndv.astype(float)

    # Initialize weight matrices
    IrIndvW = np.zeros((m, n, c, camnum), dtype=float)
    indvW = np.zeros((m, n, c, camnum), dtype=float)

    # Process each camera's rectification
    for k in range(camnum):
        ir = IrIndv[..., k]  # Extract individual rectification

        # Set pixels with [0, 0, 0] to NaN (assume all channels match)
        mask = np.all(ir == 0, axis=-1)
        ir[mask] = np.nan

        # Compute binary mask (1 = nan, 0 = valid pixels)
        binI = np.isnan(ir[..., 0])

        # Compute distance transform (distance to nearest NaN pixel)
        D = scipy.ndimage.distance_transform_edt(~binI)

        # Compute weighting function
        if np.isinf(D).all():  # If all pixels are valid, uniform weighting
            W = np.ones_like(D)
        else:
            W = D / np.nanmax(D)  # Normalize distances

        # Expand weight matrix to match image channels
        W = np.repeat(W[..., np.newaxis], c, axis=-1)
        W[np.isnan(W)] = 0  # Set NaNs to 0

        # Apply weights to rectified image
        IrIndvW[..., k] = np.nan_to_num(ir * W)
        indvW[..., k] = W

    # Compute weighted average, avoiding division by zero
    weight_sum = np.sum(indvW, axis=-1)
    weight_sum[weight_sum == 0] = np.nan  # Prevent division by zero

    Ir = np.sum(IrIndvW, axis=-1) / weight_sum  # Weighted average
    
    # Convert to uint8 and handle NaN cases
    Ir = np.nan_to_num(Ir).astype(np.uint8)

    return Ir

def local_transform_points(xo, yo, ang, flag, xin, yin):
    """
    Transforms between local World Coordinates and Geographical World Coordinates.
    
    :param xo (float): X coordinate of the local origin in geographical coordinates.
    :param yo (float): Y coordinate of the local origin in geographical coordinates.
    :param ang (float): Angle (in radians) between the local X axis and the geographical X axis.
    :param flag (int): Direction of transformation (1: Geo to Local, 0: Local to Geo).
    :param xin (ndarray): Input X coordinates (Local or Geo depending on transformation direction).
    :param yin (ndarray): Input Y coordinates (Local or Geo depending on transformation direction).
    
    :return: 
    - xout (ndarray): Transformed X coordinates.
    - yout (ndarray): Transformed Y coordinates.
    """
    
    if flag == 1:
        # Transform from Geographical to Local
        easp = xin - xo
        norp = yin - yo
        xout = easp * np.cos(ang) + norp * np.sin(ang)
        yout = norp * np.cos(ang) - easp * np.sin(ang)
    else:
        # Transform from Local to Geographical
        xgeo = xin * np.cos(ang) - yin * np.sin(ang)
        ygeo = yin * np.cos(ang) + xin * np.sin(ang)
        xout = xgeo + xo
        yout = ygeo + yo
    
    return xout, yout

def filter_regions(img):
    """
    Filters regions in an image by finding and keeping only the largest contour.

    :param img (ndarray): Input image to process, in BGR color format.
    
    :return: 
    - img (ndarray): The input image with all regions outside the largest contour set to black.
    - contours (list): A list of contours found in the binary mask of the image.
    """
    # Load image and convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    threshold = 10  # Adjust if necessary
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No contours found

    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create an empty mask (same shape as binary_mask) filled with zeros
    largest_contour_mask = np.zeros_like(binary_mask)

    # Draw only the largest contour
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
    img[largest_contour_mask == 0] = [0,0,0] 
    return img, contours

def prompt_for_directory(prompt_message):
    """
    Opens a dialog to prompt the user for selecting a directory, or allows manual input.

    :param prompt_message (str): The message to display in the dialog prompting the user to select a directory.
    
    :return: 
    - str: The path of the selected directory or manually entered path.
    """
    try:
        Tk().withdraw()  # Hide the root Tk window
        folder = filedialog.askdirectory(title=prompt_message)
        if folder:
            return folder
    except Exception as e:
        print(f"Error opening folder dialog: {e}")

    # Fallback to manual input if the dialog fails
    return input(f"{prompt_message} (Enter path manually): ").strip()
