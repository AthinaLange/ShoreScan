"""
utils_runup.py

This module provides functions for computing runup statistics and getting TWL from the USGS TWL&CC forecast.

"""
import glob
import json
import os
import time
from datetime import datetime
from math import cos, degrees, radians
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
import pandas as pd
import requests
import scipy.io
import scipy.signal
from geopy.distance import geodesic
from plotly.graph_objects import go
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import yaml

import utils_CIRN

## ------------------- define transects -------------------------
def define_UV_transects(productsPath, intrinsics, extrinsics, pixsaveDir=None, yaml_filename="transects.yaml", pix_filename="output.pix"):
    """
    Processes transect data to generate corresponding pixel coordinates (UV) from world coordinates (XYZ) and saves the results.

    :params productsPath: Path to JSON file with dictionary containing transect data or a list of dictionaries.
    :params intrinsics: (dict) Camera intrinsic parameters.
    :params extrinsics: (dict) Camera extrinsic parameters.
    :params saveDir: (str, optional) Directory to save output files. Defaults to the current working directory.
    :params yaml_filename: (str, optional) Name of the YAML file to save the structured data. Default is "transects.yaml".
    :params pix_filename: (str, optional) Name of the PIX file to save filtered UV coordinates. Default is "output.pix".

    :return (dict) A dictionary containing processed transect data with Easting, Northing, Elevation, U, and V values.
    """
    # Determine the correct transect type
    if not productsPath.endswith('.json'):
        productsPath = os.path.join(productsPath, "products.json")  # Append 'products.json' if it's a directory

    with open(productsPath, "r") as file:
        products = json.load(file)
    if isinstance(products, list):
        products_x = next((item for item in products if item.get("type") == "xTransect"), None)
    # If products is already a dictionary, assume it's the desired item
    elif isinstance(products, dict) and products.get("type") == "xTransect":
        products_x = products
    else:
        products_x = None  # If neither, return None

    if not yaml_filename.endswith('.yaml'):
        yaml_filename = os.path.join(yaml_filename, ".yaml") 

    if not pix_filename.endswith('.pix'):
        pix_filename = os.path.join(pix_filename, ".pix")

    # Get coordinates for the transect
    output_x = utils_CIRN.get_xy_coords(products_x)

    all_UV_filtered = []  # To store concatenated valid UV values
    data_to_save = {}

    for key, data in output_x.items():
        xyz = data['xyz']
        
        # Convert XYZ to UV coordinates
        UVd, flag = utils_CIRN.xyz_to_dist_uv(intrinsics, extrinsics, xyz)
        if UVd.shape[0] == 2 and UVd.shape[1] != 2:
            UVd = UVd.T  # Ensure shape is (#, 2)
        
        # Filter UV values that are within image bounds
        valid_mask = (UVd[:, 0] >= 0) & (UVd[:, 0] < intrinsics["NU"]) & \
                     (UVd[:, 1] >= 0) & (UVd[:, 1] < intrinsics["NV"])
        UV_filtered = UVd[valid_mask].astype(int)
        xyz_filtered = np.around(xyz[valid_mask], decimals=3)

        # Remove sporadic points based on gradient changes
        if UV_filtered.size > 0:
            max_U = np.argmax(np.abs(np.gradient(np.abs(np.gradient(UV_filtered[:, 0])))))
            max_V = np.argmax(np.abs(np.gradient(np.abs(np.gradient(UV_filtered[:, 1])))))
            change_loc = np.max([max_U, max_V])

            if change_loc < len(UV_filtered) / 2:
                UV_filtered = np.delete(UV_filtered, np.s_[:change_loc + 1], axis=0)
                xyz_filtered = np.delete(xyz_filtered, np.s_[:change_loc + 1], axis=0)
            else:
                UV_filtered = np.delete(UV_filtered, np.s_[change_loc:], axis=0)
                xyz_filtered = np.delete(xyz_filtered, np.s_[change_loc:], axis=0)

        # Extract individual components
        x_vals = xyz_filtered[:, 0]
        y_vals = xyz_filtered[:, 1]
        z_vals = xyz_filtered[:, 2]
        u_vals = UV_filtered[:, 0]
        v_vals = UV_filtered[:, 1]

        # Store processed data
        data_to_save[key] = {
            "Easting": ",".join(map(str, x_vals)),
            "Northing": ",".join(map(str, y_vals)),
            "Elevation": ",".join(map(str, z_vals)),
            "U": ",".join(map(str, u_vals)),
            "V": ",".join(map(str, v_vals))
        }
        
        # Store valid UV data for concatenation
        all_UV_filtered.append(UV_filtered)
    
    # Ensure save directory is defined
    if not pixsaveDir:
        pixsaveDir = os.getcwd()
    os.makedirs(pixsaveDir, exist_ok=True)

    # Save to YAML file
    with open(os.path.join(pixsaveDir, yaml_filename), "w") as yaml_file:
        yaml.dump(data_to_save, yaml_file, default_flow_style=False)
        
    # Save concatenated UV data to PIX file
    if all_UV_filtered:
        all_UV_filtered = np.vstack(all_UV_filtered)
        np.savetxt(os.path.join(pixsaveDir, pix_filename), all_UV_filtered[:, :2], fmt="%i", delimiter=" ")
    
    return data_to_save

## ------------------- Process ras image -------------------------
 
def get_camera_settings(file_path = None):
    """
    Load camera settings from a JSON file and convert date strings back to datetime objects.
    The JSON file should be named `camera_settings.json` and structured as follows:
    
    ```json
    {
        "SITE_ID": {  # Each site (e.g., "CACO03", "CACO04") is a key
            "CHANNEL_ID": {  # Each channel (e.g., "c1", "c2") under the site
                "reverse_flag": bool,  # Indicates if pix coordinates should be reversed (false: offshore to onshore)
                "coordinate_files": {  # Maps time ranges to file paths
                    "START_TIME|END_TIME": "file_path"  # Time range (ISO 8601 format) mapped to a file
                }
            }
        }
    }
    Example JSON:
    {
        "CACO03": {
            "c1": {
                "reverse_flag": false,
                "coordinate_files": {
                    "2024-09-20T15:29:00|2024-11-08T00:00:00": "data/c1_timestack_20240920.pix",
                    "2024-11-08T00:00:00|2025-01-24T00:00:00": "data/c1_timestack_20241107.pix"
                }
            },
            "c2": {
                "reverse_flag": false,
                "coordinate_files": {
                    "2024-09-20T15:29:00|2024-11-08T00:00:00": "data/c2_timestack_20240920.pix",
                    "2024-11-08T00:00:00|2025-01-24T00:00:00": "data/c2_timestack_20241107.pix"
                }
            }
        }
    }
    :param file_path: (str, optional) Path to `camera_settings.json`. If None, prompts user to select a directory.
    :return: (dict) Parsed camera settings with datetime keys.
    :raises FileNotFoundError: If the file is not found.
    """
    
    # If no file path is provided, prompt the user to select a directory
    if file_path is None:
        Tk().withdraw()  # Hide the root Tkinter window
        directory = filedialog.askdirectory(title="Select the directory containing camera_settings.json")
        
        if not directory:  # If the user cancels, exit
            raise FileNotFoundError("No directory selected.")
        
        file_path = os.path.join(directory, "camera_settings.json")  # Look for the JSON file in the selected directory

    # Ensure the file exists
    if not os.path.exists(file_path):
        try:
            Tk().withdraw()  # Hide the root Tkinter window
            directory = filedialog.askdirectory(title="Select the directory containing camera_settings.json")
            file_path = os.path.join(directory, "camera_settings.json")
        except:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load JSON file
    with open(file_path, "r") as f:
        loaded_data = json.load(f)

    # Convert string timestamps back to tuple of datetime objects
    for camera in loaded_data.values():
        for cam in camera.values():
            cam["coordinate_files"] = {
                tuple(datetime.fromisoformat(ts) for ts in k.split("|")): v
                for k, v in cam["coordinate_files"].items()
            }
    
    return loaded_data  # Return the entire structure

def get_coordinate_file_for_timestamp(camera_config, timestamp):
    """
    Retrieve the appropriate coordinate file for a given camera and timestamp.

    The function checks which `.pix` coordinate file corresponds to the given timestamp 
    based on the time ranges specified in `camera_config`.

    :param camera_config: (dict) Camera configuration containing time-based coordinate files.
    :param timestamp: (datetime) The timestamp of the image.
    :return: (str or None) Path to the coordinate file, or None if no suitable file is found.
    """
    # Pull coordinate files dictionary from camera_config.
    coordinate_files = camera_config.get('coordinate_files', {})

    # find appropriate .pix coordinate file based on timestamp of image. 
    for file_time, coord_file in coordinate_files.items():
        if timestamp >= file_time[0] and timestamp <= file_time[1]:
            return coord_file  # Return the coordinate file for this time range

    return None

def split_tiff(config = None, imageDir = None, yamlDir = None, camera_settingsPath = None, threshold = 100):
    """
    Splits `.ras.tiff` images into transects based on coordinate files and saves them. 
    It checks if a transect file already exists and skips the image processing if it does.

    Path inputs either provided directly or in config.

    :param config: (dict or str, optional) Configuration file or dictionary.
    :param imageDir: (str, optional) Directory containing image files.
    :param yamlDir: (str, optional) Directory for saving YAML files.
    :param camera_settingsPath: (str, optional) Path to camera settings JSON file.
    :param threshold: (int, optional) Threshold for detecting jumps in the coordinates (default: 100).
    """
    if config:
        if isinstance(config, str) and os.path.exists(config):
            with open(config, "r") as f:
                config = json.load(f)

    imageDir = config.get("imageDir", imageDir)
    yamlDir = config.get("yamlDir", yamlDir)
    ras_images = glob.glob(f"{imageDir}/*ras.tiff")
    camera_settings = get_camera_settings(file_path = config.get("camera_settingsPath", camera_settingsPath))
    # Get products_grid
    productsPath = config.get("productsPath", {})
    if not productsPath.endswith('.json'):
        productsPath = os.path.join(productsPath, "products.json")  # Append 'products.json' if it's a directory

    with open(productsPath, "r") as file:
        products = json.load(file)
    if isinstance(products, list):
        products_x = next((item for item in products if item.get("type") == "xTransect"), None)
    elif isinstance(products, dict) and products.get("type") == "xTransect":
        products_x = products
    else:
        products_x = None  # If neither, return None

    if products_x:
        y_def = np.sort(products_x['y'])[::-1]
    else: 
        y_def = []

    for img_path in ras_images:
        print(img_path)
        filename = Path(img_path).stem
        camera_config = camera_settings.get(filename.split('.')[6], {}).get(filename.split('.')[7], {})
        reverse_flag = camera_config.get('reverse_flag', False)
        timestamp = datetime.fromtimestamp(int(filename.split('.')[0]))
        
        coordinate_file_path = get_coordinate_file_for_timestamp(camera_config, timestamp)
        print(coordinate_file_path)

        base_name = os.path.splitext(os.path.basename(coordinate_file_path))[0]
        if len(y_def) > 0:
            expected_transect_file = os.path.join(yamlDir, f"{base_name}_transect{y_def[0]}.yaml")
        else:
            expected_transect_file = os.path.join(yamlDir, f"{base_name}_transect0.yaml")

        # Check if transect file exists
        if os.path.exists(expected_transect_file):
            print(f"Transect files already exist for {coordinate_file_path}, skipping.")
        else:
            print(f"Splitting coordinates from {coordinate_file_path} into transects...")
            split_and_save_pixcoordinates(coordinate_file_path, yamlDir, reverse_data=reverse_flag, threshold=100, y_def = y_def)

        try:
            print(f"Processing and splitting image: {img_path}")
            process_and_split_image(image_path = img_path, coordinate_file_path=coordinate_file_path, reverse_data = reverse_flag, threshold = threshold, output_path = os.path.join(os.path.dirname(img_path), 'split_timestacks'), y_def = y_def)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
        
def split_and_save_pixcoordinates(coordinate_file_path, yamlDir, reverse_data = False, threshold = 100, y_def = []):
    """
    Splits `.pix` coordinate data into individual transects based on detected jumps and saves them.

    :param coordinate_file_path: (str) Path to the `.pix` coordinate file.
    :param output_path: (str) Directory where transect files will be saved.
    :param reverse_data: (bool, optional) Reverse order of data points (default: False).
    :param threshold: (int, optional) Distance threshold to detect jumps (default: 100).
    :param y_def: (ndarray, optional) Along-shore transect locations - otherwise will enumerate.
    :raises ValueError: If the file cannot be loaded.
    """
    # Load the data (x, y points)
    data = pd.read_csv(coordinate_file_path, sep=r'\s+', header=None, names=['x', 'y'])
    
    # If reverse_data flag = True 
    if reverse_data:
        data = data.iloc[::-1].reset_index(drop=True)
        print("Data reversed.")
    
    # Compute distances between neighboring points to detect jumps
    distances = np.sqrt(np.diff(data['x'])**2 + np.diff(data['y'])**2)
    jumps = np.where(distances > threshold)[0] + 1
    transects = [data.iloc[start:end] for start, end in zip([0] + list(jumps), list(jumps) + [len(data)])]
    
    base_name = Path(coordinate_file_path).stem
    
    # Save each transect in a YAML file
    for i, transect in enumerate(transects):
        # Convert U and V into comma-separated strings
        transect_dict = {
            'U': ', '.join(map(str, transect['x'].tolist())),
            'V': ', '.join(map(str, transect['y'].tolist()))
        }
        if len(y_def) > 0 and i < len(y_def):
            transect_file = os.path.join(yamlDir, f"{base_name}_transect{y_def[i]}.yaml")
        else:
            transect_file = os.path.join(yamlDir, f"{base_name}_transect{i}.yaml")

        with open(transect_file, 'w') as file:
            yaml.dump(transect_dict, file, default_flow_style=False)

            # Manually add comments at the beginning
            file.write("# U: row pixel location\n")
            file.write("# V: column pixel location\n")
        print(f"Saved {transect_file}")

def process_and_split_image(image_path, coordinate_file_path, output_path = None, reverse_data = False, plot = False, threshold = 100, y_def = []):
    """
    Process and split a `.ras` timestack image based on transect data.

    :param image_path: (str) Path to the image file.
    :param coordinate_file_path: (str) Path to the `.pix` coordinate file.
    :param output_path: (str, optional) Directory where cropped images will be saved.
    :param reverse_data: (bool, optional) Reverse the order of data points (default: False).
    :param plot: (bool, optional) Plot transects and fitted lines (default: False).
    :param threshold: (int, optional) Threshold to detect jumps (default: 100).
    :param y_def: (ndarray, optional) Along-shore transect locations - otherwise will enumerate.
    :raises ValueError: If the image cannot be loaded.
    """
    if not output_path:
        output_path = os.path.dirname(image_path)  # Use the image's directory instead

    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_height, image_width = image.shape[:2]
    print(f"Image size (height, width): {image_height}, {image_width}")

    # Step 2: Load the data (x, y points of the lines)
    data = pd.read_csv(coordinate_file_path, sep=r'\s+', header=None)
    data.columns = ['x', 'y']
    if reverse_data:
        data = data.iloc[::-1].reset_index(drop=True)
        print("Data reversed.")

    # Step 3: Compute distances between neighboring points to detect jumps
    distances = np.sqrt(np.diff(data['x'])**2 + np.diff(data['y'])**2)
    jumps = np.where(distances > threshold)[0] + 1
    transects = [data.iloc[start:end] for start, end in zip([0] + list(jumps), list(jumps) + [len(data)])]

    # Optional: Plot transects and fitted lines
    if plot:
        fig = go.Figure()
        for i, transect in enumerate(transects):
            fig.add_trace(go.Scatter(
                x=transect['x'], y=transect['y'], mode='markers',
                name=f'transect {i+1}', marker=dict(size=8)
            ))
            model = LinearRegression()
            model.fit(transect['x'].values.reshape(-1, 1), transect['y'])
            x_range = np.linspace(transect['x'].min(), transect['x'].max(), 100)
            y_range = model.predict(x_range.reshape(-1, 1))
            fig.add_trace(go.Scatter(
                x=x_range, y=y_range, mode='lines',
                name=f'Fit {i+1}', line=dict(dash='dash', width=2)
            ))
        fig.show()

    # Step 4: Save cropped images
    os.makedirs(output_path, exist_ok=True)
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    shift = 0
    for i, transect in enumerate(transects):
        transect_width = len(transect)
        cropped_image = image[:, shift:shift + transect_width]
        shift += transect_width
        if cropped_image.size > 0:
            if len(y_def) > 0 and i < len(y_def):
                output_file = os.path.join(output_path, f"{base_image_name}_transect{y_def[i]}.png")
            else:
                output_file = os.path.join(output_path, f"{base_image_name}_transect{i}.png")
            cv2.imwrite(output_file, cropped_image)
            print(f"Saved {output_file}")



## ----------- TWL forecast -----------
def bounding_box(lat, lon, distance_m):
    """
    Calculate the bounding box for a given lat, lon, and distance in meters.
    :param lat: Latitude of the center point (in decimal degrees).
    :param lon: Longitude of the center point (in decimal degrees).
    :param distance_m: Distance in meters for the bounding box.
    :return: Dictionary with min_lat, max_lat, min_lon, max_lon.
    """
    # Earth's radius in meters
    R = 6378137

    # Offsets in radians
    d_lat = distance_m / R
    d_lon = distance_m / (R * cos(radians(lat)))

    # Calculate the bounding box
    min_lat = lat - degrees(d_lat)
    max_lat = lat + degrees(d_lat)
    min_lon = lon - degrees(d_lon)
    max_lon = lon + degrees(d_lon)

    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon
    }

def get_valid_latitude():
    """
    Prompt the user to enter a valid latitude value between -90 and 90.
    :return: A valid latitude value as a float.
    """
    while True:
        try:
            lat = float(input("Enter latitude: "))
            if -90 <= lat <= 90:
                return lat
            else:
                print("Invalid latitude. Please enter a value between -90 and 90.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for latitude.")

def get_valid_longitude():
    """
    Prompt the user to enter a valid longitude value between -180 and 180.
    :return: A valid longitude value as a float.
    """
    while True:
        try:
            lon = float(input("Enter longitude: "))
            if -180 <= lon <= 180:
                return lon
            else:
                print("Invalid longitude. Please enter a value between -180 and 180.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for longitude.")

def validate_lat_lon(lat, lon):
    """
    Validate that the provided lat and lon are within valid ranges.
    :param lat: Latitude value.
    :param lon: Longitude value.
    :return: Validated lat and lon.
    """
    if lat is not None:
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90.")
    if lon is not None:
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180.")
    return lat, lon

def make_request_with_retry(url, params=None):
    """
    Make a GET request and handle rate limiting.
    If the API rate limit is exceeded (HTTP 429), the function will wait for the duration provided
    in the 'X-Retry-After' header before retrying the request.
    """
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 429:
            # Get the retry delay from the X-Retry-After header
            retry_after = response.headers.get("X-Retry-After", "1")  # Default to 1 second if not set
            try:
                retry_after = float(retry_after)  # Convert the value to a float
            except ValueError:
                retry_after = 1  # In case the value cannot be converted, use a default value of 1 second
            print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            response.raise_for_status()  # If status is not 429, raise the usual exceptions
            return response.json()

def fetch_water_level_data(region_choice=None, site_id=None, lat=None, lon=None, timestamp=None,  distance_m=500,  save_folder=None):
    """
    Fetch water level data from the USGS TWL&CC viewer API based on coordinates and timestamp.
    If any required parameter is missing, the function prompts for it interactively.
    :param lat: Latitude of the location (optional if site_id is provided).
    :param lon: Longitude of the location (optional if site_id is provided).
    :param timestamp: Timestamp in "YYYY-MM-DD HH:MM:SS" format.
    :param region_choice: The selected region index.
    :param distance_m: Search radius in meters (default: 500).
    :param site_id: The site ID (optional if lat/lon are provided).
    :param save_folder: The folder where the data should be saved (optional).
    :return: Dictionary of water level data.
    """
    BASE_URL = "https://coastal.er.usgs.gov/hurricanes/research/twlviewer/api/"

    try:
        # Regions
        regions_url = f"{BASE_URL}regions"
        regions = make_request_with_retry(regions_url)
        if region_choice is None:
            print("Available Regions:")
            for i, region in enumerate(regions):
                print(f"{i + 1}: {region['fullName']}")
            region_choice = int(input("Select a region by number: "))
            if region_choice < 1 or region_choice > len(regions):
                raise ValueError("Invalid region choice.")
        selected_region = regions[region_choice - 1]
        region_id = selected_region['id']
        region_name = selected_region['fullName']

        # Timestamp
        if timestamp is None:
            timestamp = input("Enter timestamp (YYYY-MM-DD): ")
        user_time = datetime.strptime(timestamp, "%Y-%m-%d")

        if site_id is not None:
            file_name = f"{region_name.replace(' ', '_')}_{site_id}_{user_time.date()}.json"
            file_path = os.path.join(save_folder, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                water_level = json.load(f)
        else:

            # Forecasts
            forecasts_url = f"{BASE_URL}regions/{region_id}/forecasts?date={user_time.date()}&time=00:00:00"
            forecasts = make_request_with_retry(forecasts_url)
            forecast_id = forecasts[0]['id']

            # Site ID
            if site_id is None:
                # Prompt for lat, lon, and timestamp if not provided
                if lat is None:
                    lat = get_valid_latitude()
                if lon is None:
                    lon = get_valid_longitude()
                
                bbox = bounding_box(lat, lon, distance_m)

                # Validate lat and lon
                lat, lon = validate_lat_lon(lat, lon)
                params = {
                    "siteLatitude": [f"gte_{bbox['min_lat']}", f"lte_{bbox['max_lat']}"],
                    "siteLongitude": [f"gte_{bbox['min_lon']}", f"lte_{bbox['max_lon']}"],
                }

                # Retrieve site data
                sites_url = f"{BASE_URL}regions/{region_id}/forecasts/{forecast_id}/sites?pageSize=100"
                sites = make_request_with_retry(sites_url, params=params)

                # Find the closest site
                closest_site = None
                min_distance = 20  # Minimum distance threshold in kilometers
                for site in sites:
                    site_coords = (site['siteLatitude'], site['siteLongitude'])
                    distance = geodesic((lat, lon), site_coords).kilometers
                    if distance < min_distance:
                        closest_site = site
                        min_distance = distance

                if not closest_site:
                    raise ValueError("No suitable site found near the provided coordinates.")

                site_id = closest_site['id']
            
                    # Determine file save path
            if save_folder is None:
                save_folder = "data_wl"
            
            os.makedirs(save_folder, exist_ok=True)

            file_name = f"{region_name.replace(' ', '_')}_{site_id}_{user_time.date()}.json"
            file_path = os.path.join(save_folder, file_name)
            
            print(file_path)

            # Fetch file
            water_levels_url = f"{BASE_URL}regions/{region_id}/forecasts/{forecast_id}/sites/{site_id}/waterLevels"
            data = make_request_with_retry(water_levels_url)
            water_level = {key: [] for key in data[0].keys()}

            for entry in data:
                for key, value in entry.items():
                    try:
                        water_level[key].append(float(value))
                    except ValueError:
                        water_level[key].append(value)

            # Save the data to the file
            with open(file_path, 'w') as json_file:
                json.dump(water_level, json_file, indent=4)

            print(f"Water level data saved to {file_path}")

        water_level['dateTime'] = pd.to_datetime(water_level['dateTime'])

        return water_level
    except Exception as e:
        print(f"Error processing {timestamp}: {e}")

## ----------- Runup statistics -----------
def runupStatistics_CHI(eta, t_sec, nperseg, f_lims=np.array([0.004, 0.04, 0.35]), grd=None):
    """
    Computes statistical parameters of runup from a water level time series.

    :param eta (ndarray): Runup time series (tide should be subtracted).
    :param t_sec (ndarray): Time array in seconds corresponding to eta.
    :param nperseg (int): Window length for spectral analysis.
    :param f_lims (ndarray): Array of frequency limits [min IG, IG/SS split, max SS] (default: [0.004, 0.04, 0.35]).
    :param grd (dict, optional): Bathymetric data along transect. Should contain:
        - 'x': Local cross-shore coordinates (in meters),
        - 'z': Elevation (in meters).

    :return: 
    dict: Dictionary of computed runup statistics, including:
        - "setup": Mean water level.
        - "eta2": 2% exceedance level from full time series.
        - "R2": 2% exceedance level from runup peaks.
        - "S2": Swash height (R2 - setup).
        - "Ts", "Tr": Swash and runup periods (time/ # of peaks).
        - "Tp": Peak wave period.
        - "Ss": Significant swash (4 * sqrt(var(eta))).
        - "Ssin": Significant incident swash.
        - "Ssig": Significant infragravity swash.
        - "Sst": Total significant swash.
        - "f", "S": Frequency array and spectral density.
        - "beta_S2006": Beach slope from ±2 std elevation band (if grd provided).
        - "beta_Z": Beach slope from total elevation range (if grd provided).

    Notes:
        - Modified from Dave's version of runupFullStats... and subroutines.
        - Hardcoded:
            - nbins = 20; number of bins for calculating CDF
            - MinPeakProminence = np.std(eta)/3; Minimum prominence for find peaks
            - distance=3; minimum distance between peaks
    """
    
    print('computing runup stats')
    RUstats = {
        "R2": [],
        "Tr": [],
        "S2": [],
        "Ts": [],
        "setup": [],
        "eta2": [],
        "f": [],
        "S": [],
        "Tp": [],
        "Ss": [],
        "Ssin": [],
        "Ssig": [],
        "Sst": [],
        "beta_S2006":[],
        "beta_Z": []
    }

    # sample rate
    dt = np.mean(np.diff(t_sec))

    # Minimum prominence for find peaks
    MinPeakProminence = np.std(eta) / 3

    # Number of bins for cummulative distribution function
    nbins = 20

    ## Setup
    setup = np.mean(eta)
    RUstats["setup"] = setup

    # 2# exceedence value of eta i.e. full time series not peaks
    # Do this using CDF.
    # Define bins.
    c, binCenters = CDF_by_bins(eta, nbins)
    id = np.argmin(np.abs(c - 0.98))
    eta2 = np.interp(0.98, c, binCenters)
    RUstats["eta2"] = eta2

    ## Runup and swash
    # Swash is defined as the range of values between successive zero
    # crossings.
    # find runup peaks
    peaks, _ = find_peaks(eta, prominence=MinPeakProminence, distance=3)
    # plt.plot(eta)
    # plt.plot(peaks, eta[peaks], "x")
    # plt.xlim([0,100])
    # plt.show()

    # 2# exceedence value runup and swash
    R = eta[peaks]
    c, binCenters = CDF_by_bins(R, nbins)
    R2 = np.interp(0.98, c, binCenters)
    # 2% excceedence
    RUstats["R2"] = R2
    RUstats["S2"] = R2 - np.mean(eta)

    # swash period (same as runup period?!)
    RUstats["Ts"] = t_sec[-1] / len(peaks)
    RUstats["Tr"] = t_sec[-1] / len(peaks)

    # Power Spectral Density
    f, S = scipy.signal.welch(
        eta, fs=1 / dt, window="hann", nperseg=nperseg, noverlap=nperseg * (3 / 4)
    )
    RUstats["f"] = f
    RUstats["S"] = S

    # Peak frequency
    fp = f[np.argmax(S)]
    # Peak period.
    Tp = 1 / fp
    RUstats["Tp"] = Tp

    # significant swash = Ss
    Ss = 4 * np.sqrt(np.std(eta))
    RUstats["Ss"] = Ss
    # incident and IG band significant swash = Ssin & Ssig
    df = np.mean(np.diff(f))

    inc = (f >= f_lims[1]) & (f < f_lims[2])
    ig = (f >= f_lims[0]) & (f < f_lims[1])
    # incident
    varin = sum(S[inc]) * df
    # IG
    varig = sum(S[ig]) * df
    # Total
    vartot = sum(S[ig | inc]) * df

    RUstats["Ssin"] = 4 * np.sqrt(varin)
    RUstats["Ssig"] = 4 * np.sqrt(varig)
    RUstats["Sst"] = 4 * np.sqrt(vartot)


    if grd is not None:
        stdEta = np.std(eta)

        maxEta = setup + 2 * stdEta
        minEta = setup - 2 * stdEta
        botRange = np.where((grd['z'] >= minEta) & (grd['z'] <= maxEta))[0]
        fitvars = np.polyfit(grd['x'][botRange], grd['z'][botRange], 1)
        beta = fitvars[0]
        RUstats["beta_S2006"] = beta

        maxEta = np.max(eta)
        minEta = np.min(eta)
        botRange = np.where((grd['z'] >= minEta) & (grd['z'] <= maxEta))[0]
        fitvars = np.polyfit(grd['x'][botRange], grd['z'][botRange], 1)
        beta = fitvars[0]
        RUstats["beta_Z"] = beta

    return RUstats

def CDF_by_bins(x, nbins):
    """
    Computes the empirical cumulative distribution function (CDF) of a dataset using histogram binning.

    :param x (ndarray): Input 1D array of data.
    :param nbins (int): Number of bins to use in histogram.

    :return: 
    - c (ndarray): Cumulative distribution values.
    - binCenters (ndarray): Centers of the bins used for the CDF.
    """
    # Do this using CDF.
    # Define bins.
    binWidth = (np.max(x) - np.min(x)) / nbins
    bins = np.arange(np.min(x), np.max(x), binWidth)
    binCenters = bins + 0.5 * binWidth
    c = np.zeros(len(bins))  # cdf
    for ii in np.arange(1, len(bins)).reshape(-1):
        c[ii] = sum(x < bins[ii]) / len(x)
    return c, binCenters

def runup_stats_CHI(Ibp, exc_value=None):
    """
    Computes the 2% runup (R2) from instantaneous beach position using the zero-downcrossing method.

    :param Ibp (xarray.DataArray): Instantaneous vertical beach position (in meters relative to still water level).
    :param exc_value (float, optional): Outlier threshold (exclude values with abs(Ibp) > exc_value). Default is None.

    :return: 
    xarray.DataArray: Scalar containing the 2% runup value (R2), with units in meters and metadata attributes.

    @author: rfonsecadasilva
    """
    if exc_value:
        bad_data = (
            xr.apply_ufunc(np.isnan, Ibp.where(lambda x: np.abs(x) > exc_value)).sum()
            / Ibp.size
            * 100
        ).values.item()
        if bad_data != 100:
            print(f"{100-bad_data:.1f} % of runup data is corrupted")
        Ibp = Ibp.where(lambda x: np.abs(x) < exc_value, drop=True)
    time_cross = Ibp.where(
        ((Ibp.shift(t=1) - Ibp.mean(dim="t")) * (Ibp - Ibp.mean(dim="t")) < 0)
        & (Ibp - Ibp.mean(dim="t") > 0),
        drop=True,
    )  # calculate crossing points for zero-downcrossing on swash

    R2 = (
        Ibp.groupby_bins("t", time_cross.t).max(dim="t").quantile(0.98).values
    )  # calculate 2% runup,dim="t_bins"
    print(R2)
    R2 = xr.DataArray(R2, dims=())

    R2.attrs["standard_name"], R2.attrs["long_name"], R2.attrs["units"] = (
        "2% runup",
        "2% runup",
        "m",
    )
    return R2
