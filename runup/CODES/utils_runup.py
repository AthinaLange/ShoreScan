'''
utils_runup.py
This module provides functions for defining the U,V coordinates for an ARGUS-style camera and extracting individual transect images from a ras image.

Dependencies:
-------------
- os
- json
- yaml
- opencv (cv2)
- utm
- glob
- datetime
- pandas
- numpy
- pathlib
- tkinter
- plotly
- sklearn

'''
# Custom Modules
from RunUpTimeseriesFunctions_CHI import *
from utils_CIRN import *

# External Libraries
import os
import json
import yaml
import cv2
import utm
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from tkinter import Tk, filedialog
from tkinter.filedialog import askdirectory
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

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
    output_x = get_xy_coords(products_x)

    all_UV_filtered = []  # To store concatenated valid UV values
    data_to_save = {}

    for key, data in output_x.items():
        xyz = data['xyz']
        
        # Convert XYZ to UV coordinates
        UVd, flag = xyz_to_dist_uv(intrinsics, extrinsics, xyz)
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
    for img_path in ras_images:
        print(img_path)
        filename = Path(img_path).stem
        camera_config = camera_settings.get(filename.split('.')[6], {}).get(filename.split('.')[7], {})
        reverse_flag = camera_config.get('reverse_flag', False)
        timestamp = datetime.fromtimestamp(int(filename.split('.')[0]))
        
        coordinate_file_path = get_coordinate_file_for_timestamp(camera_config, timestamp)
        print(coordinate_file_path)
        base_name = os.path.splitext(os.path.basename(coordinate_file_path))[0]
        expected_transect_file = os.path.join(yamlDir, f"{base_name}_transect0.yaml")
        # Check if transect file exists
        if os.path.exists(expected_transect_file):
            print(f"Transect files already exist for {coordinate_file_path}, skipping.")
        else:
            print(f"Splitting coordinates from {coordinate_file_path} into transects...")
            split_and_save_pixcoordinates(coordinate_file_path, yamlDir, reverse_data=reverse_flag, threshold=100)

        try:
            print(f"Processing and splitting image: {img_path}")
            process_and_split_image(image_path = img_path, coordinate_file_path=coordinate_file_path, reverse_data = reverse_flag, threshold = threshold)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
        
def split_and_save_pixcoordinates(coordinate_file_path, yamlDir, reverse_data = False, threshold = 100):
    """
    Splits `.pix` coordinate data into individual transects based on detected jumps and saves them.

    :param coordinate_file_path: (str) Path to the `.pix` coordinate file.
    :param output_path: (str) Directory where transect files will be saved.
    :param reverse_data: (bool, optional) Reverse order of data points (default: False).
    :param threshold: (int, optional) Distance threshold to detect jumps (default: 100).
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

        transect_file = os.path.join(yamlDir, f"{base_name}_transect{i}.yaml")
        
        with open(transect_file, 'w') as file:
            yaml.dump(transect_dict, file, default_flow_style=False)

            # Manually add comments at the beginning
            file.write("# U: row pixel location\n")
            file.write("# V: column pixel location\n")
        print(f"Saved {transect_file}")

def process_and_split_image(image_path, coordinate_file_path, output_path = None, reverse_data = False, plot = False, threshold = 100):
    """
    Process and split a `.ras` timestack image based on transect data.

    :param image_path: (str) Path to the image file.
    :param coordinate_file_path: (str) Path to the `.pix` coordinate file.
    :param output_path: (str, optional) Directory where cropped images will be saved.
    :param reverse_data: (bool, optional) Reverse the order of data points (default: False).
    :param plot: (bool, optional) Plot transects and fitted lines (default: False).
    :param threshold: (int, optional) Threshold to detect jumps (default: 100).
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
            output_file = os.path.join(output_path, f"{base_image_name}_transect{i}.png")
            cv2.imwrite(output_file, cropped_image)
            print(f"Saved {output_file}")







## ------------------- timestack runup stuff -------------------

def get_runup(config = None, timestackDir = None, overlayDir = None, site_settingsPath = None):
    """
    Processes extracted runup data, computes shoreline elevations, runup statistics and saves to netCDF.

    :param config: (dict, optional) Configuration dictionary containing paths and values.
    :param timestackDir: (str, optional) Path to the timestacks folder.
    :param overlayDir: (str, optional) Output folder for overlays (default: 'timestackDir/overlays').
    :param site_settingsPath: (str, optional) Path to site settings file.

    :return: None
    """
    config = config or {}  # Ensure config is always a dictionary

    timestackDir = config.get("timestackDir", timestackDir)
    overlayDir = config.get("overlayDir", overlayDir)
    site_settingsPath = config.get("site_settingsPath", site_settingsPath)

    # Ensure timestackDir is set
    if not timestackDir:
        raise ValueError("A timestack directory must be provided either as an argument or in the config.")
    overlayDir = overlayDir or os.path.join(timestackDir, "overlays")

    datastore = ImageDatastore(timestackDir)
    datastore.load_images()
    # Remove all non-RAS image types
    for img_type in ['bright', 'dark', 'var', 'timex', 'snap']:
        datastore.remove_images_by_type(img_type)

    # Filter unmatched images and extract metadata
    datastore.filter_unmatched_images(overlayDir)
    datastore.extract_metadata()
    datastore.image_stats()

    for site, cameras in datastore.images.items():
        for camera, years in cameras.items():
            for year, months in years.items():
                for month, days in months.items():
                    for day, times in days.items():
                        for time, images_by_type in times.items():
                            for image_type, images in images_by_type.items():
                                for img in images:
                                    print(img['path'])

                                    # Extract metadata
                                    metadata = img.get('metadata', {})
                                    
                                    # Extract 'extrinsics' and 'intrinsics'
                                    extrinsics = metadata.get("extrinsics", None)
                                    intrinsics = metadata.get("intrinsics", None)
                                    
                                    # Extract 'transect' and its 'U' and 'V'
                                    transect = metadata.get("transect", {})
                                    U = np.array(transect.get("U", []))
                                    V = np.array(transect.get("V", []))
                                    transect_date = transect.get("transect_date", [])

                                    # Get corresponding runup data
                                    runup_file = os.path.join(overlayDir, f'runup_{Path(img["path"]).stem}.txt')
                                    if os.path.isfile(runup_file):
                                        runup_data = np.genfromtxt(runup_file, delimiter=None)  # Auto-detect delimiter
                                        # Skip files that are **entirely NaN**
                                        if np.all(np.isnan(runup_data)):
                                            print(f"Skipping {runup_file} (all values are NaN)")
                                            continue  # Don't plot this file
                                        else:
                                            h_runup_id = np.round(len(U) - runup_data).astype(int)  
                                    else:
                                        print(f"No runup data exists for {Path(img['path']).stem}. Please rerun extract_runup.")
                                        continue

                                    # Extract date and transect number
                                    fileName = Path(img['path']).stem.split('.')
                                    date = fileName[0]
                                    try:
                                        transectNum = fileName[8].split('_')[1][-1]
                                    except IndexError:
                                        print(f"Could not determine transect number for {img['path']}")
                                        continue

                                    # Get DEM and compute shoreline elevation
                                    DEM = 0#get_DEM(date = date, site = site, camNum = camera, transectNum = transectNum)
                                    xyz = np.zeros((3,3))#dist_uv_to_xyz(intrinsics, extrinsics, np.column_stack((U, V)), 'z', DEM['z'])
                                    Hrunup = U#xyz[h_runup_id, 0]
                                    Zrunup = V#xyz[h_runup_id, 2]
                                    
                                    # Store results in netCDF
                                    sites = get_site_settings(file_path = site_settingsPath)
                                    site_dict = sites.get(site, {})
                                    if "siteInfo" not in site_dict:
                                        print(f"Missing site info for {site}. Skipping.")
                                        continue
                                    site_dict["siteInfo"]["camNum"] = camera
                                    site_dict["siteInfo"]["rNumber"] = transectNum
                                    write_netCDF(site_dict, img['path'], U, V, transect_date, xyz,  Hrunup, Zrunup)
                                
    return 

def get_DEM(date, site, camNum, transectNum):
    """
     df = pd.read_csv('Marconi 2024-10-23.csv')
    t = np.linspace(0, 1, len(df))
    t_fine = np.linspace(0, 1, 50)
    E_fine = interp1d(t, df['Easting'], kind='cubic')(t_fine)
    N_fine = interp1d(t, df['Northing'], kind='cubic')(t_fine)
    Z_fine = interp1d(t, df['Elevation'], kind='cubic')(t_fine)
    
    F = griddata((E_fine, N_fine), Z_fine, (xyz[:, 0], xyz[:, 1]), method='linear')
    DEM_z = pd.Series(F).rolling(window=25, win_type='gaussian', center=True).mean(std=2).values
    """
    return




## ------------------- Save data --------------------
def write_netCDF(site, img_path, U, V, transect_date, xyz, Hrunup, Zrunup):
        
    #history = f"{datetime.now(timezone.utc).isoformat()} Using Python version {sys.version}, netCDF4 version {nc.__version__}, NumPy version {np.__version__}"

    # ----------------- U,V -----------------
    U = U
    V = V

    # ----------------- Camera -----------------
    ECamera = float(yamlData['x'])
    NCamera = float(yamlData['y'])
    zCamera = float(yamlData['z'])

    # ----------------- UTM -----------------
    UTM_E = xyz[:,0]
    UTM_N = xyz[:,1]


    # ----------------- Lat/Lon -----------------
    lat, lon = utm.to_latlon(UTM_E, UTM_N, int(site["siteInfo"]["utmZone"][0:2]), site["siteInfo"]["utmZone"][2:3])

    # ----------------- z -----------------
    z = xyz[:,2]

    # ----------------- X,Y -----------------
    X = U
    Y = V
    # ----------------- Time  -----------------
    t_sec = np.around(np.arange(0, sample_period_length*60, 1/sample_frequency_Hz)[:RAW.shape[0]], decimals=3) # timeseries in seconds
    T = np.array([fileDatetime + timedelta(seconds = t) for t in t_sec])

    # ----------------- Runup -----------------
    Hrunup = np.array(Hrunup)
    Zrunup = np.array(Zrunup)

    # ----------------- TWL stats -----------------
    # window length with 8 windows
    #window_length = np.floor(len(twl) / 8).astype(int) - 1
    #window_length = window_length.item()
    nperseg = 1 #<< window_length.bit_length() # next power of two for length
    
    
    try:
        TWLstats = runupStatistics_CHI(Zrunup, t_sec, nperseg, site["sampling"]["freqLimits"][1])
    except Exception as e:
        print(f"Error running runup(): {e}")  # Print error message
        TWLstats = {}  # Fallback to an empty dictionary to prevent crashes


    return

