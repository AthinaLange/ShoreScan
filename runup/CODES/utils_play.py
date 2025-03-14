'''
utils Module
==========================
This module provides utility functions for ShoreScan, a system designed to process and analyze shoreline imagery data.

It includes functionality for:
- Creating videos from images
- Checking processed images
- Removing processed images
- Processing images with machine learning models

Modules Imported:
-----------------
- **Workflow & Datastore Modules**: Handles workflow processes and data storage.
- **Custom Modules**: Imports ShorelineWorkflow, TimestackWorkflow, ImageDatastore, and ShorelineDatastore.
- **External Libraries**: Provides various utilities such as image processing, data handling, and numerical computing.
- **GUI & File Dialogs**: Supports user interaction for file selection and visualization.
- **Plotting & Visualization**: Enables graphical representation of image and data outputs.
- **EXIF Functions**: Handles image metadata extraction and manipulation.
- **Machine Learning & Computer Vision**: Implements models for image segmentation and analysis.
- **Miscellaneous**: Includes additional utilities for file management and computations.

Dependencies:
-------------
- OpenCV (cv2)
- NumPy
- Pandas
- SciPy
- PIL (Pillow)
- Matplotlib
- Plotly
- Torch (PyTorch)
- scikit-image
- exiftool
- piexif
- tkinter (for GUI file dialogs)
- pvlib (for solar positioning calculations)

Example Usage:
--------------
>>> from utils import create_video_from_images
>>> create_video_from_images('/path/to/images', 'output.mp4')

'''

# Workflow & Datastore Modules
# Custom Modules
from c_Datastore import ImageDatastore
from RunUpTimeseriesFunctions_CHI import *

# External Libraries
import sys
import os
import re
import json
import logging
import pytz
import exiftool
import cv2
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import scipy.interpolate
import scipy.signal
import utm
import yaml
from datetime import datetime, timezone, timedelta
from time import time
from zoneinfo import ZoneInfo
from pathlib import Path
from PIL import Image
import glob
import piexif
from skimage.morphology import skeletonize

# GUI & File Dialogs
from tkinter import Tk, filedialog
from tkinter.filedialog import askdirectory

# Plotting & Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# Machine Learning & Computer Vision
import torch
from segment_anything import SamPredictor, sam_model_registry
from sklearn.linear_model import LinearRegression

# Miscellaneous
from collections import defaultdict
import shutil
from pvlib import solarposition
from math import radians
import csv


## ------------------- shoreline stuff -------------------
def create_video_from_images(datastore, video_name="output_video.mp4", frame_rate=30, image_type='timex', camera=None, site=None):
    """
    Create a video from images in the datastore with optional filtering by image type, camera, and site.
    
    :param datastore: (ImageDatastore) The ImageDatastore object containing the images to be processed.
    :param video_name: (str, optional) The name of the output video file. Default is "output_video.mp4".
    :param frame_rate: (int, optional) The frame rate for the video (frames per second). Default is 30.
    :param image_type: (str, optional) The type of images to include in the video (e.g., 'bright', 'snap'). Default is 'timex'.
    :param camera: (str, optional) The camera identifier to filter by (e.g., 'CACO03'). If None, process all cameras. Default is None.
    :param site: (str, optional) The site identifier to filter by. If None, process all sites. Default is None.

    :return: None
    
    :raises ValueError: If no images match the filter criteria.
    """
    
    # Get all images by type from the datastore, optionally filter by site and camera
    images_metadata = datastore.get_image_metadata_by_type([image_type], site=site, camera=camera)
    
    if not images_metadata:
        print(f"No images found for image_type: {image_type} with the specified filters.")
        return
    
    # Sort images by timestamp (ensure they are in chronological order)
    images_metadata.sort(key=lambda x: x['timestamp'])
    
    # Read the first image to obtain the video dimensions
    first_image = cv2.imread(images_metadata[0]['path'])
    height, width, layers = first_image.shape
    
    # Define video parameters and initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
    out = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))  # Using specified frame rate
    
    # Loop through the images and add them to the video
    for img_metadata in images_metadata:
        img_path = img_metadata['path']
        image = cv2.imread(img_path)
        
        # Add text overlay with additional metadata (site, camera, date, and time)
        site = img_metadata['site']
        camera = img_metadata['camera']
        month = img_metadata['month']
        day = img_metadata['day']
        time = img_metadata['time']
        
        text = f"Site: {site} | Camera: {camera} | {month}-{day} {time}"
        
        # Add text overlay to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, height - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame to the video
        out.write(image)
    
    # Release the video writer
    out.release()
    print(f"Video created successfully: {video_name}")

def check_processed_images(image_metadata, pt_dir, output_dir):
    """
    Check if an image has been processed by verifying the existence of its corresponding shoreline point and plot files.

    :param image_metadata: (dict) A dictionary containing metadata of the image to check.
        Expected keys include 'site', 'camera', 'year', 'month', 'day', 'time', and 'image_type'.
    :param pt_dir: (str) Directory path where shoreline point files (e.g., '.txt' files) are stored.
    :param output_dir: (str) Directory path where shoreline output files (e.g., '.png' plots) are stored.

    :return: (bool) True if the image has been processed (i.e., both the point and plot files exist), 
             False otherwise.
    """
    site = image_metadata['site']
    camera = image_metadata['camera']
    year = image_metadata['year']
    month = image_metadata['month']
    day = image_metadata['day']
    time = image_metadata['time']
    image_type = image_metadata['image_type']

    textname = f"{site}_{camera}_{year}_{month}_{day}_{time}_{image_type}_shoreline_points.txt"
    plotname = f"{site}_{camera}_{year}_{month}_{day}_{time}_{image_type}_shoreline_plot.png"

    pt_path = os.path.join(pt_dir, plotname)
    output_path = os.path.join(output_dir, textname)

    return os.path.exists(pt_path) and os.path.exists(output_path)

def remove_processed_images(datastore, pt_dir, output_dir):
    """
    Remove images from the datastore that have already been processed (i.e., their corresponding 
    shoreline point and plot files exist).

    :param datastore: (ImageDatastore) An instance of the ImageDatastore class containing image metadata.
    :param pt_dir: (str) Directory path for shoreline point files.
    :param output_dir: (str) Directory path for shoreline output files.
    :return: None
    """
    for site, cameras in list(datastore.images.items()):
        for camera, years in list(cameras.items()):
            for year, months in list(years.items()):
                for month, days in list(months.items()):
                    for day, times in list(days.items()):
                        for time, images_by_type in list(times.items()):
                            for img_type, img_list in list(images_by_type.items()):
                                filtered_images = [
                                    img for img in img_list
                                    if not check_processed_images(img, pt_dir, output_dir)
                                ]
                                # Update datastore
                                datastore.images[site][camera][year][month][day][time][img_type] = filtered_images

                                # Remove empty entries
                                if not filtered_images:
                                    del datastore.images[site][camera][year][month][day][time][img_type]

def traverse_datastore(datastore, image_type=None, process_function=None):
    """
    Generic traversal function for the ImageDatastore structure.

    :param datastore: (ImageDatastore) The datastore to traverse.
    :param image_type: (str, optional) The image type to filter by. If None, processes all image types.
    :param process_function: (callable) A function to apply at each image metadata during traversal.
                              The function signature should be:
                              process_function(site, camera, year, month, day, time, image_type, image_metadata)
    """
    for site, cameras in datastore.images.items():
        for camera, years in cameras.items():
            for year, months in years.items():
                for month, days in months.items():
                    for day, times in days.items():
                        for time, images_by_type in times.items():
                            if image_type:
                                # Filter by specific image type
                                if image_type in images_by_type:
                                    for image_metadata in images_by_type[image_type]:
                                        process_function(
                                            site, camera, year, month, day, time, image_type, image_metadata
                                        )
                            else:
                                # Process all image types
                                for img_type, image_list in images_by_type.items():
                                    for image_metadata in image_list:
                                        process_function(
                                            site, camera, year, month, day, time, img_type, image_metadata
                                        )

def process_images(datastore, img_type, shoreline_datastore, make_intermediate_plots):
    """
    Process images of a specified type and perform shoreline analysis using the ShorelineWorkflow class.

    :param datastore: (ImageDatastore) An instance of the ImageDatastore class containing image metadata.
    :param img_type: (str) The type of image to process (e.g., 'bright', 'timex').
    :param shoreline_datastore: (ShorelineDatastore) An instance of the ShorelineDatastore class to store shoreline analysis results.
    :param make_intermediate_plots: (bool) Flag indicating whether intermediate plots should be generated during processing.
    """
    for site, cameras in datastore.images.items():
        for camera, years in cameras.items():
            for year, months in years.items():
                for month, days in months.items():
                    for day, times in days.items():
                        for time, images_by_type in times.items():
                            if img_type in images_by_type:
                                for img_metadata in images_by_type[img_type]:
                                    img_path = img_metadata['path']
                                    print(f"Processing image: {img_path}")
                                    try:
                                        workflow = ShorelineWorkflow(
                                            image_path=img_path,
                                            image_type=img_type,
                                            shoreline_datastore=shoreline_datastore,
                                            make_intermediate_plots=make_intermediate_plots,
                                        )
                                        workflow.process()
                                        shoreline_datastore.save_shoreline_coords_to_file(
                                            site = site,
                                            camera = camera,
                                            year = year,
                                            month = month,
                                            day = day,
                                            time = time,
                                            image_type = img_type,
                                            outputDir = "shoreline_output",
                                        )
                                    except Exception as e:
                                        print(f"Error processing {img_path}: {e}")

## ------------------- timestack runup stuff -------------------
def prep_ras_images(config = None, split_tiff = True):
    """
    Prepares '.ras' images for timestack processing.
    Split '.ras' images into individual timestacks and embed metadata. 
    
    :param split_tiff: (bool) Flag on whether to split tiff or just insert metadata
    :param config: (dict, optional) Dictionary with predefined directories. Defaults to None.
    :return: (str) Path to the timestacks folder.
    """
    config = config or {}  # Ensure config is always a dictionary
    split_tiff = config.get("split_tiff", split_tiff)  # Get from config, fallback to function argument

    # Load camera settings
    camera_settings = get_camera_settings(file_path = config.get("camera_settingsDir"))

    # Determine timestack directory (explicit argument takes precedence)
    timestackDir = config.get("timestackDir")
    yamlDir = config.get("yamlDir", os.path.join(os.getcwd(), 'YAML'))
    jsonDir = config.get("jsonDir", os.path.join(os.getcwd(), 'JSON'))


    
    # Ask user to select output folder for timestacks
    if not timestackDir:
        Tk().withdraw()
        timestackDir = askdirectory(title="Select individual timestack folder.")
    if split_tiff:
        datastore = ImageDatastore(config.get("rawDir"))
        datastore.load_images()
        # Remove all non-RAS image types
        for img_type in ['bright', 'dark', 'var', 'timex']:
            datastore.remove_images_by_type(img_type)
        datastore.image_stats()
        # Split images into individual timestacks
        split_ras_images(datastore = datastore, timestackDir = timestackDir, yamlDir = yamlDir, camera_settings = camera_settings)
        
    
    # Dictionary to store site timestack directories
    site_dirs = {}
    # Loop through site names from camera_settings.json
    for site_name in camera_settings.keys():
        formatted_site_name = re.sub(r'([a-zA-Z])(\d)', r'\1-\2', site_name)
        site_dirs[site_name] = Site(formatted_site_name, site_name)
        site_dirs[site_name].timestackDir = timestackDir
        
    # embed metadata into image  
    datastore = ImageDatastore(timestackDir)
    datastore.load_images()  
    for site_name in datastore.images.keys(): 
        process_timestacks(site_dirs[site_name], jsonDir = jsonDir, yamlDir = yamlDir) 

    return timestackDir

def extract_runup(config = None, timestackDir = None, grayscaleDir = None, overlayDir = None, runup_val = 0, rundown_val = -1.5):
    """
    Extracts shoreline runup from timestacks using SegFormer.
    Assumes grayscale images in 'timestackDir/grayscale'.

    :param config: (dict, optional) Configuration dictionary containing paths and values.
    :param timestackDir: (str) Path to the timestacks folder.
    :param grayscaleDir: (str) Path to grayscale folder. If None: default to timestackDir/grayscale. 
    :param overlayDir: (str, optional) Output folder for overlays (default: 'timestackDir/overlays').
    :param runup_val: (float, optional) Runup threshold (default: 0).
    :param rundown_val: (float, optional) Rundown threshold (default: -1.5).

    :return: None
    """
    config = config or {}  # Ensure config is always a dictionary

    # Override parameters with config values if provided
    timestackDir = config.get("timestackDir", timestackDir)
    grayscaleDir = config.get("grayscaleDir", grayscaleDir)
    overlayDir = config.get("overlayDir", overlayDir)
    runup_val = config.get("runup_val", runup_val)
    rundown_val = config.get("rundown_val", rundown_val)
    # Ensure a timestack directory is set
    if not timestackDir:
        raise ValueError("A timestack directory must be provided either as an argument or in the config.")

    # Convert timestacks into grayscale format for SegFormer
    flip_and_save_grayscale(inputDir = timestackDir)  
    
    # Get SegFormer weights and model
    weightDir = config.get("segformerWeightsDir") if config else askdirectory(title="Select folder for SegFormer weights.")
    model = config.get("model_option", 'SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5')

    weights = os.path.join(weightDir, model)

    if not os.path.exists(weights):
        raise FileNotFoundError(f"SegFormer weights not found in {weightDir}")

    # Run SegFormer model
    grayscaleDir = grayscaleDir or os.path.join(timestackDir, "grayscale")
    segformerCodeDir = config.get("segformerCodeDir", os.getcwd())
    subprocess.call(['python', os.path.join(segformerCodeDir, 'segformer.py'), grayscaleDir, weights])

    # Overlay runup line and extract shoreline
    overlayDir = overlayDir or os.path.join(timestackDir, "overlays")
    flip_and_overlay_segment_lines_from_npz(
            rawDir = timestackDir, 
            segDir = os.path.join(grayscaleDir, 'meta'), 
            outputDir = overlayDir, 
            rundown_val = rundown_val, runup_val = runup_val)

    return

def get_runup(config = None, timestackDir = None, overlayDir = None, site_settingsDir = None):
    """
    Processes extracted runup data, computes shoreline elevations, runup statistics and saves to netCDF.

    :param config: (dict, optional) Configuration dictionary containing paths and values.
    :param timestackDir: (str, optional) Path to the timestacks folder.
    :param overlayDir: (str, optional) Output folder for overlays (default: 'timestackDir/overlays').
    :param site_settingsDir: (str, optional) Path to site settings file.

    :return: None
    """
    config = config or {}  # Ensure config is always a dictionary

    timestackDir = config.get("timestackDir", timestackDir)
    overlayDir = config.get("overlayDir", overlayDir)
    site_settingsDir = config.get("site_settingsDir", site_settingsDir)

    # Ensure timestackDir is set
    if not timestackDir:
        raise ValueError("A timestack directory must be provided either as an argument or in the config.")
    overlayDir = overlayDir or os.path.join(timestackDir, "overlays")

    datastore = ImageDatastore(timestackDir)
    datastore.load_images()
    # Remove all non-RAS image types
    for img_type in ['bright', 'dark', 'var', 'timex']:
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
                                    sites = get_site_settings(file_path = site_settingsDir)
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

## ------------------- prep tiff images stuff ------------------- commented
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

def process_and_split_image(image_path, coordinate_file_path, output_path, reverse_data = False, plot = False, threshold = 100):
    """
    Process and split a `.ras` timestack image based on transect data.

    :param image_path: (str) Path to the image file.
    :param coordinate_file_path: (str) Path to the `.pix` coordinate file.
    :param output_path: (str) Directory where cropped images will be saved.
    :param reverse_data: (bool, optional) Reverse the order of data points (default: False).
    :param plot: (bool, optional) Plot transects and fitted lines (default: False).
    :param threshold: (int, optional) Threshold to detect jumps (default: 100).
    :raises ValueError: If the image cannot be loaded.
    """
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

def split_ras_images(datastore, timestackDir, yamlDir, camera_settings, threshold = 100):
    """
    Processes `.ras` images from the datastore by identifying the correct `.pix` coordinate file 
    and splitting the image into individual transects.

    Steps:
    1. Retrieve the correct camera settings based on site, camera, and timestamp.
    2. Identify the appropriate `.pix` coordinate file.
    3. Split the `.ras` image into individual transects.

    :param datastore: (ImageDatastore) Object containing the `.ras` images categorized by site, camera, and timestamp.
    :param timestackDir: (str) Path where individual timestacks will be saved.
    :param camera_settings: (dict) Dictionary containing camera configurations with `reverse_flag`
                            and coordinate files for specific time ranges.
    :param threshold: (int, optional) Distance threshold to detect transect breaks (default: 100).
    """
    for site, cameras in datastore.images.items():
        # Get correct camera_settings for the current site.
        site_config = camera_settings.get(site, {})
        for camera, years in cameras.items():
            for year, months in years.items():
                for month, days in months.items():
                    for day, times in days.items():
                        for time, images_by_type in times.items():
                            for image_type, image_list in images_by_type.items():
                                # only processed .ras types
                                if image_type == 'ras':
                                    for image_metadata in image_list:
                                        img_path = image_metadata['path']
                                        print(f"Processing image: {img_path}")
                                        # Step 1: Pull .pix coordinates

                                        # Determine the reverse_data_flag and coordinate file based on camera and date/time
                                        camera_config = site_config.get(camera, {})
                                        reverse_flag = camera_config.get('reverse_flag', False)

                                        # Get the Unix timestamp from image_metadata and convert it to a datetime object
                                        timestamp = image_metadata['timestamp']  

                                        # Convert timestamp if it's a string
                                        if isinstance(timestamp, str):
                                            try:
                                                timestamp = int(timestamp)
                                            except ValueError:
                                                print(f"Invalid timestamp format for image {img_path}: {timestamp}")
                                                continue  # Skip this image if conversion fails

                                        # Pull the correct .pix coordinate file based on image timestamp
                                        timestamp = datetime.fromtimestamp(timestamp)  
                                        coordinate_file_path = get_coordinate_file_for_timestamp(camera_config, timestamp)

                                        # Ensure the coordinate file exists before proceeding
                                        if not coordinate_file_path or not os.path.exists(coordinate_file_path):
                                            print(f"Coordinate file not found for {img_path}. Skipping.")
                                            continue
                                        
                                        # Check if coordinate file has already been split
                                        base_name = os.path.splitext(os.path.basename(coordinate_file_path))[0]
                                        expected_transect_file = os.path.join(yamlDir, f"{base_name}_transect0.txt")

                                        if os.path.exists(expected_transect_file):
                                            print(f"Transect files already exist for {coordinate_file_path}, skipping.")
                                        else:
                                            print(f"Splitting coordinates from {coordinate_file_path} into transects...")
                                            split_and_save_pixcoordinates(
                                                coordinate_file_path, 
                                                yamlDir, 
                                                reverse_data=reverse_flag, 
                                                threshold=threshold
                                            )
                                        
                                        # Step 2: Split .ras image into individual transects
                                        try:
                                            print("Processing and splitting image...")
                                            process_and_split_image(img_path, coordinate_file_path, timestackDir, reverse_data=reverse_flag, threshold=threshold)
                                        except Exception as e:
                                            print(f"Error processing image {img_path}: {e}")

## ------------------- Exif stuff -------------------
def process_timestacks(site, jsonDir, yamlDir):
    """
    Processes timestack images by extracting metadata, 
    generating EXIF tags, and writing them back to the images.

    :param site: Site object containing directory paths and calibration information.
    """
    print(site.timestackDir)
    timestack_list = glob.glob(site.timestackDir + '/*.png')

    for image in timestack_list:
        filename = image.split('/')[-1]
        filenameElements = filename.split('.')
        camNum = int(filenameElements[7][1])
        product_type = filenameElements[8][:3]
        epoch_time = filenameElements[0]
        fileDatetime = datetime.fromtimestamp(int(epoch_time), tz=pytz.utc)
        transect = filenameElements[8][-1]

        # Load JSON Tags
        json_tags = loadMultiCamJson(image, site, jsonDir)

        # Prepare EXIF Data
        tags = json_tags['All']
        productTags = json_tags['Products'][product_type]
        for tag in productTags:
            productTags[tag] = productTags[tag].replace("[insert_creation_datetime]", str(fileDatetime) + ' UTC')
        tags.update(productTags)

        calibDict = createCalibDict_timestacks(site, yamlDir=yamlDir, fileDatetime=fileDatetime, camNum=camNum, transect=transect)
        yamlData = eval(calibDict['data'])
        tags.update({
            'GPSDateStamp': yamlData['calibration_date'],
            'GPSTimeStamp': '00:00:00',
            'UserComment': calibDict,
            'PreservedFilename': filename,
            'ImageDescription': tags['Description'],
            'DateTimeOriginal': str(fileDatetime)[:19],
            'DateTime': str(datetime.now())
        })
        # Write EXIF Data
        with exiftool.ExifToolHelper(executable='exiftool') as et:
            et.set_tags([image], tags=tags, params=["-P", "-overwrite_original"])
            keywordList = tags['Keywords'].split(', ')
            for k, keyword in enumerate(keywordList):
                et.execute(f'-keywords{"+" if k else ""}={keyword}', image, "-P", "-overwrite_original")

def loadMultiCamJson(image, site, jsonDir, year=None):
    '''
    Special case function for loading JSON for sites with multiple cameras.
    JSON tags loaded depend on the camera number.
    Inputs:
        image (string) - image filepath
        site (site object) - CoastCam site object
    Outputs:
        jsonTags (dict) - dictionary of json tags and values
    '''
    
    filename = image.split('/')[-1]
    #different than image products
    filenameElements = filename.split('_')
    #strcutured different for IO and EO
    if 'GMT' in filename:
        filenameElements = filename.split('.')
        camNum = int(filenameElements[7][1])
    else:
        camNum = int(filenameElements[1][1])
        
    if year != None:
        jsonFile = f'{jsonDir}/{site.nName}EXIF_c{camNum}_{year}.json'
    else:
        jsonFile = f'{jsonDir}/{site.nName}EXIF_c{camNum}.json'
        
    with open(jsonFile) as f:
        jsonTags = json.load(f)

    return jsonTags

def createCalibDict_timestacks(site, yamlDir, fileDatetime=None, camNum=None, transect=None):
    """
    Create the dictionary object containing the camera calibration parameters
    (extrinsics, intrinsics, local origin, metadata) for a given site. Read in data
    from YAML files. If necessary, for sites like Madeira Beach, also specify a datetime
    to select the appropriate extrinsic calibration and metadata YAML files. Each dictionary
    also contains descriptions of what each variable is as well as a note describing the dictionary.
    Inputs:
        site (Site) - Python Site object
        fileDatetime (datetime) - Optional. Datetime for a specific file to determine what
                                  YAML file to select for extrinsics and metadata.
        camNum (int) - optional string specifying camera number to use when searching for YAML file
    Outputs:
        calibDict (dictionary) - dictionary object of variables and their descriptors
    """

    calibDict = {}
    calibDict[
        "Note"
    ] = 'The following nested dictionary contains the cameras intrinsic and extrinsic calibration parameters. Intrinsic camera parameters use the Brown distortion model (Brown, D.C., 1971, "Close-Range Camera Calibration", Photogrammetric Engineering.), using the Camera Calibration Toolbox for Matlab by Jean-Yves Bouguet (https://doi.org/10.22002/D1.20164). Extrinsic calibration parameters were computed using the CIRN Quantitative Coastal Imaging Toolbox (Bruder, B. L. and Brodie, K. L., 2020, "CIRN Quantitative Coastal Imaging Toolbox", SoftwareX, 12 (10052), https://doi.org/10.1016/j.softx.2020.100582.).'
  
    # there are multiple calibrations from different dates
    if fileDatetime != None:
        extrinsics, extrinsicFile = getExtrinsics(site, yamlDir=yamlDir, fileDatetime=fileDatetime, camNum=camNum)
        metadata, metadataFile = getMetadata(site, yamlDir=yamlDir, fileDatetime=fileDatetime, camNum=camNum)
        intrinsics, intrinsicFile = getIntrinsics(site, yamlDir=yamlDir, fileDatetime=fileDatetime, camNum=camNum)
        transect, transectFile = getTransect(site, yamlDir=yamlDir, fileDatetime=fileDatetime, camNum=camNum, transectNum=transect)

    else:
        extrinsics, extrinsicFile = getExtrinsics(site, yamlDir=yamlDir, camNum=camNum)
        metadata, metadataFile = getMetadata(site, yamlDir=yamlDir, camNum=camNum)
        intrinsics, intrinsicFile = getIntrinsics(site, yamlDir=yamlDir, camNum=camNum)
        transect, transectFile = getTransect(site, yamlDir=yamlDir, camNum=camNum, transectNum=transect)

    #localOrigin, localOriginFile = getLocalOrigin(site)

    # read data fields into one dictionary
    data_fields = {}
    yaml_list = []
    yaml_list.append(extrinsics)
    yaml_list.append(intrinsics)
    yaml_list.append(metadata)
    #yaml_list.append(localOrigin)
    for dictionary in yaml_list:
        for key in dictionary:
            data_fields[key] = str(dictionary[key])

    # Read comments from YAML file into one dictionary
    comment_fields = {}
    comment_list = []
    intr_comments = readYAMLcomments(intrinsicFile)
    #localOrg_comments = readYAMLcomments(localOriginFile)
    extr_comments = readYAMLcomments(extrinsicFile)
    metadata_comments = readYAMLcomments(metadataFile)
    transect_comments = readYAMLcomments(transectFile)
    comment_list.append(extr_comments)
    comment_list.append(intr_comments)
    comment_list.append(metadata_comments)
    comment_list.append(transect_comments)
    #comment_list.append(localOrg_comments)
    for dictionary in comment_list:
        for key in dictionary:
            comment_fields[key] = str(dictionary[key])

    # Read transects from YAML file into one dictionary
    transect_fields = {}
    transect_list = []
    transect_list.append(transect)
    #comment_list.append(localOrg_comments)
    for dictionary in transect_list:
        for key in dictionary:
            transect_fields[key] = str(dictionary[key])

    calibDict["data"] = str(data_fields)
    calibDict["descriptions"] = str(comment_fields)
    calibDict["transect"] = str(transect_fields)
    return calibDict

def getExtrinsics(site, yamlDir, fileDatetime=None, camNum=None):
    """
    Get the camera calibration extrinsics from a YAML file, returned as a Python dictionary,
    for a CoastCam site. If a datetime is specified, find the extrinsics YAML file with the
    closest calibration date (in the filename) following that datetime.
    Inputs:
        site (Site) - CoastCam Site Python object
        fileDatetime (datetime) - datetime used for searching YAML files for closest time
        camNum (int) - optional string specifying camera number to use when searching for YAML file
    Outputs:
        extrinsics (dictionary) - dictionary of extrinsic calibration variables
        extrinsicFile (string) - name of extrinsic calibration YAML file
    """
    
    # check if camera number is specified
    try:
        hasCamNum = hasattr(site, "camNum")
        camNum = site.camNum
    except AttributeError:
        pass

    #if no cam num found for site, see if cam num was given as input
    if not hasCamNum:
        if camNum != None:
            hasCamNum = True

    # datetime specified, find closest date following calibration date
    if fileDatetime != None:
        # search for yaml files specific to site
        yamlList = glob.glob(os.path.join(yamlDir, f"{site.nName}*_EO.yaml"))
        yamlDateList = uniqueYamlDates(yamlList)

        # get closest calibration date
        ind = getClosestPreviousDateIndex(yamlDateList, fileDatetime)
        closestDate = yamlDateList[ind]
        closestDate = datetime.strftime(closestDate, "%Y%m%d")

        if hasCamNum:
            # ex: {yamlDir}/madbeach_c1_20170217_EO.yaml
            extrinsicFile = os.path.join(yamlDir, (f"{site.nName}_c{str(camNum)}_{closestDate}_EO.yaml"))
        else:
            # ex: {yamlDir}/madbeach_20170217_EO.yaml
            extrinsicFile = os.path.join(yamlDir, f"{site.nName}_{closestDate}_EO.yaml")

    else:
        # ex: {yamlDir}/madbeach_EO.yaml
        extrinsicFile = os.path.join(yamlDir, f"{site.nName}_EO.yaml")

    extrinsics = yaml2dict(extrinsicFile)
    print(extrinsicFile)
    return extrinsics, extrinsicFile

def getIntrinsics(site, yamlDir, fileDatetime=None, camNum=None):
    """
    Get the camera calibration intrinsics from a YAML file, returned as a Python dictionary,
    for a CoastCam site. If a datetime is specified, find the intrinsics YAML file with the
    closest calibration date (in the filename) following that datetime.
    Inputs:
        site (Site) - CoastCam Site Python object
        fileDatetime (datetime) - datetime used for searching YAML files for closest time
        camNum (int) - optional string specifying camera number to use when searching for YAML file
    Outputs:
        intrinsics (dictionary) - dictionary of intrinsic calibration variables
        intrinsicFile (string) - name of intrinsic calibration YAML file
    """

    # check if camera number is specified
    try:
        hasCamNum = hasattr(site, "camNum")
        camNum = site.camNum
    except AttributeError:
        pass

    #if no cam num found for site, see if cam num was given as input
    if not hasCamNum:
        if camNum != None:
            hasCamNum = True

    # datetime specified, find closest date following calibration date
    if fileDatetime != None:
        # search for yaml files specific to site
        yamlList = glob.glob(os.path.join(yamlDir, f"{site.nName}*_IO.yaml"))
        yamlDateList = uniqueYamlDates(yamlList)

        # get closest calibration date
        ind = getClosestPreviousDateIndex(yamlDateList, fileDatetime)
        closestDate = yamlDateList[ind]
        closestDate = datetime.strftime(closestDate, "%Y%m%d")

        if hasCamNum:
            # ex: {yamlDir}/madbeach_c1_20170217_IO.yaml
            intrinsicFile = os.path.join(yamlDir, (f"{site.nName}_c{str(camNum)}_{closestDate}_IO.yaml"))
        else:
            # ex: {yamlDir}/duck_20010101_IO.yaml
            intrinsicFile = os.path.join(yamlDir, f"{site.nName}_{closestDate}_IO.yaml")

    else:
        # ex: {yamlDir}/madbeach_IO.yaml
        intrinsicFile = os.path.join(yamlDir, f"{site.nName}_IO.yaml")

    intrinsics = yaml2dict(intrinsicFile)
    print(intrinsicFile)
    return intrinsics, intrinsicFile

def getMetadata(site, yamlDir, fileDatetime=None, camNum=None):
    """
    Get the camera calibration metadata from a YAML file, returned as a Python dictionary,
    for a CoastCam site. If a datetime is specified, find the metadata YAML file with the
    closest calibration date (in the filename) following that datetime.
    Inputs:
        site (Site) - CoastCam Site Python object
        fileDatetime (datetime) - datetime used for searching YAML files for closest time
        camNum (int) - optional string specifying camera number to use when searching for YAML file
    Outputs:
        metadata (dictionary) - dictionary of metadata variables
        metadataFile (string) - name of metadata YAML file
    """

    # check if camera number is specified
    try:
        hasCamNum = hasattr(site, "camNum")
        camNum = site.camNum
    except AttributeError:
        pass

    #if no cam num found for site, see if cam num was given as input
    if not hasCamNum:
        if camNum != None:
            hasCamNum = True
    # datetime specified, find closest date following calibration date
    if fileDatetime != None:
        # search for yaml files specific to site
        yamlList = glob.glob(os.path.join(yamlDir, f"{site.nName}*_metadata.yaml"))
        yamlDateList = uniqueYamlDates(yamlList)

        # get closest calibration date
        ind = getClosestPreviousDateIndex(yamlDateList, fileDatetime)
        closestDate = yamlDateList[ind]
        closestDate = datetime.strftime(closestDate, "%Y%m%d")

        if hasCamNum:
            # ex: {yamlDir}/madbeach_c1_20170217_metadata.yaml
            metadataFile = os.path.join(yamlDir, (f"{site.nName}_c{str(camNum)}_{closestDate}_metadata.yaml"))
        else:
            # ex: {yamlDir}/duck_20010101_metadata.yaml
            metadataFile = os.path.join(yamlDir, f"{site.nName}_{closestDate}_metadata.yaml")

    else:
        # ex: {yamlDir}/madbeach_metadata.yaml
        metadataFile = os.path.join(yamlDir, f"{site.nName}_metadata.yaml")

    metadata = yaml2dict(metadataFile)
    return metadata, metadataFile

def getTransect(site, yamlDir, transectNum, fileDatetime=None, camNum=None):
    """
    Get the camera calibration extrinsics from a YAML file, returned as a Python dictionary,
    for a CoastCam site. If a datetime is specified, find the extrinsics YAML file with the
    closest calibration date (in the filename) following that datetime.
    Inputs:
        site (Site) - CoastCam Site Python object
        transectNum (int) - transect number
        fileDatetime (datetime) - datetime used for searching YAML files for closest time
        camNum (int) - optional string specifying camera number to use when searching for YAML file
    Outputs:
        transect (dictionary) - dictionary of U,V coordinates
        transectFile (string) - name of U,V coordinate file
    """

    # check if camera number is specified
    try:
        hasCamNum = hasattr(site, "camNum")
        camNum = site.camNum
    except AttributeError:
        pass

    #if no cam num found for site, see if cam num was given as input
    if not hasCamNum:
        if camNum != None:
            hasCamNum = True

    # search for yaml files specific to site
    yamlList = glob.glob(os.path.join(yamlDir, f"{site.nName}_c{camNum}_*_transect*.yaml"))
    yamlDateList = uniqueYamlDates(yamlList)

    # get closest calibration date
    ind = getClosestPreviousDateIndex(yamlDateList, fileDatetime)
    closestDate = yamlDateList[ind]
    closestDate = datetime.strftime(closestDate, "%Y%m%d")

    transectFile = os.path.join(yamlDir, (f"{site.nName}_c{str(camNum)}_timestack_{closestDate}_transect{transectNum}.yaml"))

    transect = yaml2dict(transectFile)
    transect["transect_date"] = str(closestDate)
    return transect, transectFile

def getClosestPreviousDateIndex(datetimeList, currentDatetime):
    """
    Compare a list of datetimes to a single (current) datetime and return the index of the closest
    datetime in the list. The closest datetime will be a datetime less than or equal to the currentDatetime.
    Note: All datetimes must be timezone-aware.
    Inputs:
        datetimeList (datetime) - list of datetime (timezone-aware) objects
        currentDatetime (datetime) - single datetime (timezone-aware) that each datetime in the
                                     list is compared against
    Outputs:
        closestIndex (int) - index datetime in datetimeList closest to currentDatetime
    """

    # make currentDateTime TZ-naive for comparison
    currentDatetime = currentDatetime.replace(tzinfo=None)
    for k, dNum in enumerate(datetimeList):
        if dNum <= currentDatetime:
            dateDifference = abs(currentDatetime - dNum)

            if k == 0:
                closestIndex = k
                lowestDateDifference = dateDifference
            else:
                # better match for calibration date found
                if dateDifference < lowestDateDifference:
                    lowestDateDifference = dateDifference
                    closestIndex = k

        elif dNum > currentDatetime:
            dateDifference = abs(currentDatetime - dNum)

            if k == 0:
                closestIndex = k
                lowestDateDifference = dateDifference
            else:
                # better match for calibration date found
                if dateDifference < lowestDateDifference:
                    lowestDateDifference = dateDifference
                    closestIndex = k

    return closestIndex

def readYAMLcomments(yamlfile):
    """
    Read a YAML file and add its comments to a dict
    Args:
        yamlfile (str): YAML file to read
    Returns:
        comment_dict (dict): dict of YAML comments
    """
    comment_dict = {}
    with open(yamlfile, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith('#'):
                #clean up comment line and add to dictionary
                line = line.replace('#', '')
                line = line.strip()
                comment_dict[line.split()[0] + '_comment'] = '#' + line
    return comment_dict

def uniqueYamlDates(yamlList, camNum=None):
    """
    Given a list of YAML files, find a list of unique datetimes corresponding to their
    calibration dates.
    Inputs:
        yamlList (list) - list of YAML files
    Outputs:
        yamlDateList (list) - list of unique datetimes
    """

    yamlDateList = []
    for file in yamlList:
        filename = Path(file).stem
        filename = filename.replace("timestack_", "") if "timestack_" in filename else filename  # Remove 'timestack_' only if it exists
        filenameElements = filename.split("_")

        yamlDateStr = filenameElements[2]
        yamlDate = datetime(
            int(yamlDateStr[0:4]), int(yamlDateStr[4:6]), int(yamlDateStr[6:8])
        )

        if yamlDate not in yamlDateList:
            yamlDateList.append(yamlDate)

    return yamlDateList

def yaml2dict(yamlfile):
    """ Import contents of a YAML file as a dict
    Args:
        yamlfile (str): YAML file to read
    Returns:
        dictname (dict): dict interpreted from YAML file
    """
    dictname = None
    with open(yamlfile, "r") as infile:
        try:
            dictname = yaml.safe_load(infile)
        except yaml.yamlerror as exc:
            print(exc)
    return dictname

class Site:
    """
    Class to hold info related to each CoastCam site. The data and metadata stored in each instance
    will be standardized to easily create EXIF metadata and NetCDF files for each site.

    Attributes:
        siteName (string) - name of the site. Used to identify folder names in NAACH
        nName (string) - the site name used in the CIRN-formatted filename
        rNumber (string) - number representing the runup transect used to sample timestack
        camNum (int) - number representing the camera used to capture timestacks
        netcdfDir (String) - path to where the outputted NetCDF files will be stored
        jpgDir (string) - for Mark's ML data, path to jpg images of timestacks for the site
        rawDirTop (string) - path to the dir containing all folders of raw timestacks
        topoDir (string) - path to dir containing the site's topo surveys
        uvDir (string) - path to dir containing UV data

        references (string) - citations to other works used in data release
        contributors (string) - contributors to data release
        siteLocation (string) - city, state, country location of site
        dataOrigin (string) - center where data originates from
        camMake (string) - camera make
        camModel (string) - camera model
        camLens (string) - camera lens model
        timezone (string) - local timezone of site
        utmZone (string) - UTM zone (number + letter)
    """

    def __init__(self, siteName, nName):
        self.siteName = siteName
        self.nName = nName


## ------------------- SegFormer stuff -------------------
def flip_and_save_grayscale(inputDir, outputDir = None):
    """
    Flip all images in the input folder horizontally, convert them to grayscale,
    and save them in the output folder under a 'grayscale' subfolder.

    :param inputDir: (str) Path to the folder containing the images to process.
    :param outputDir: (str, optional) Path to the output folder where flipped and grayscale images will be saved.
                          If not specified, a 'grayscale' subfolder will be created inside `inputDir`.

    Notes:
        - Supports PNG, JPG, JPEG, and TIFF image formats.
        - Creates the output directory if it does not exist.
    """
    if outputDir is None:
        # Create a 'grayscale' subfolder inside the output folder
        outputDir = os.path.join(inputDir, 'grayscale')
    os.makedirs(outputDir, exist_ok=True)

    # Loop through all the images in the input folder
    for filename in os.listdir(inputDir):
        img_path = os.path.join(inputDir, filename)

        # Check if it's a valid image file
        if os.path.isfile(img_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'tiff')):
            print(f"Processing image: {filename}")

            # Load the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {filename}")
                continue

            # Flip the image horizontally
            flipped_image = cv2.flip(image, 1)

            # Convert the flipped image to grayscale
            grayscale_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY)

            # Define the output path for the grayscale image
            output_file = os.path.join(outputDir, f"{os.path.splitext(filename)[0]}_flipped_grayscale.png")

            # Save the grayscale image
            cv2.imwrite(output_file, grayscale_image)
            print(f"Saved flipped and grayscale image: {output_file}")

def flip_and_overlay_segment_lines_from_npz(rawDir, segDir, outputDir, rundown_val = -1.5, runup_val = 0):
    """
    Extracts runup lines from Segformer .npz files based on rundown and runup values,
    overlays them on timestack images, and saves both the images and extracted runup lines.

    :param rawDir: (str) Path to the folder containing the timestack images.
    :param segDir: (str) Path to the folder containing segmented .npz files.
    :param outputDir: (str) Path to the folder where the processed images and runup lines will be saved.
    :param rundown_val: (float, optional) Value used in softmax score to determine rundown phase (default: -1.5).
    :param runup_val: (float, optional) Value used in softmax score to determine runup phase (default: 0).

    :return: None

    Notes:
        - Processes only PNG and JPG timestack images.
        - Uses `get_SegGym_runup_pixel_timeseries` for runup line extraction.
        - Saves both overlay images (`overlay_***.jpg`) and extracted runup lines (`runup_***.txt`).
    """
    # Get list of timestack images and .npz files
    raw_images = glob.glob(os.path.join(rawDir, "*.png")) + glob.glob(os.path.join(rawDir, "*.jpg"))
    seg_images = glob.glob(os.path.join(segDir, "*.npz")) 

    # Ensure output directory exists
    os.makedirs(outputDir, exist_ok = True)
    
    # Create a mapping for segmented images based on their base filenames
    seg_image_map = {os.path.basename(f).replace("_flipped_grayscale_res.npz", ""): f for f in seg_images}

    # Loop through the raw images
    for raw_image_path in raw_images:
        try:
            # Get the base filename without the extension
            base_name = os.path.basename(raw_image_path).replace(".png", "")

            # Find the corresponding segmented image
            seg_image_path = seg_image_map.get(base_name)
  
            if not seg_image_path:
                print(f"Skipping {raw_image_path}: corresponding segmented image not found.")
                continue
            print(f"Running {raw_image_path}.")
            # Read the raw and segmented images
            raw_image = cv2.imread(raw_image_path)
            npz_data = np.load(seg_image_path)
            av_prob_stack = npz_data["av_prob_stack"]

            if raw_image is None or av_prob_stack is None:
                print(f"Error reading images: {raw_image_path}, {seg_image_path}")
                continue

            # Flip the segmented image horizontally
            raw_image_flipped = np.flip(raw_image, axis=1)

            # Runup extraction using softmax-like segmentation image
            Ri, _, _, _, _, _ = get_SegGym_runup_pixel_timeseries(av_prob_stack, rundown_val, runup_val, buffer = [0,2])

            # Overlay the segmented line on the raw image
            raw_rgb = cv2.cvtColor(raw_image_flipped, cv2.COLOR_BGR2RGB)
            overlay_image = raw_rgb.copy()

            # Mark the segmented line on the raw image
            # Loop through the Ri array and draw lines between consecutive points
            for i in range(1, len(Ri)):
                # Skip if the current or previous point is NaN
                if np.isnan(Ri[i-1]) or np.isnan(Ri[i]):
                    continue  # Skip this iteration if either value is NaN
                
                # Draw a line between consecutive points (i-1, Ri[i-1]) and (i, Ri[i])
                cv2.line(overlay_image, (int(Ri[i-1]),i-1), (int(Ri[i]),i), (255, 0, 0), 2)  # Blue line with thickness 2

            # Save the result
            output_path = os.path.join(outputDir, f"overlay_{base_name}.jpg")
            plt.imsave(output_path, overlay_image)
            print(f"Saved overlay image: {output_path}")

            # Save the extracted runup line to a text file
            txt_output_path = os.path.join(outputDir, f"runup_{base_name}.txt")
            with open(txt_output_path, 'w') as f:
                for value in Ri:
                    if np.isnan(value):
                        f.write("NaN\n")
                    else:
                        f.write(f"{value}\n")
        except Exception as e:
            print(f"Skipped: {e}")

def get_SegGym_runup_pixel_timeseries(softmax, rundown_val, runup_val, buffer):
    """Computes the average softmax value of pixels in the runup phase, weighted by their distance to the rundown phase.

    :param softmax: (npz data) A softmax image, where pixel values represent the probability of belonging to the foreground class.
    :param rundown_val: (float) A threshold value below which pixels are considered to be in the rundown phase.
    :param runup_val: (float) A threshold value below which pixels are considered to be in the runup phase.
    :param buffer: (list or array) Buffer size used for morphological processing in `maskExtractionCV`.
    :return: (tuple)
        - Ri (np.ndarray): A time series representing the average softmax value of pixels in the runup phase, weighted by their distance to the rundown phase.
        - Ri_0 (np.ndarray): Initial mask extraction based on the thresholded softmax image.
        - Ri_down (np.ndarray): Extracted rundown mask.
        - Ri_up (np.ndarray): Extracted runup mask.
        - rundown_peaks (np.ndarray): Indices of detected rundown peaks.
        - runup_peaks (np.ndarray): Indices of detected runup peaks.
    Notes:
        - Uses `maskExtractionCV` to extract regions of interest based on threshold values.
        - Identifies peaks in the extracted masks to distinguish rundown and runup phases.
        - If no peaks are found, returns NaN-filled arrays.
        - Uses `pchip_interpolate` to compute weights for the weighted averaging process.
    """
    Ri_0 = maskExtractionCV(softmax <= 0, buffer)
    # Find the peaks in the rundown mask.
    rundown_peaks = scipy.signal.find_peaks(
        Ri_0 - np.mean(Ri_0), prominence=np.std(Ri_0) / 2
    )[0]
    # Find the peaks in the runup mask.
    runup_peaks = scipy.signal.find_peaks(
        -Ri_0 + np.mean(Ri_0), prominence=np.std(Ri_0) / 2
    )[0]

    if len(runup_peaks) == 0 or len(rundown_peaks) == 0:
        print("No peaks found for runup or rundown phase, skipping.")
        return np.full(softmax.shape[0], np.nan), Ri_0, np.nan, np.nan, rundown_peaks, runup_peaks

    # Combine the rundown and runup peaks into a single array, sorted by peak location.
    xy_up = np.array((runup_peaks, np.ones(len(runup_peaks)))).T
    xy_down = np.array((rundown_peaks, np.zeros(len(rundown_peaks)))).T
    xy = np.vstack((xy_up, xy_down))
    xy = xy[xy[:, 0].argsort(), :]
    # Add neutral points at the beginning and end of the array.
    xy = np.vstack(([0, 0.5], xy, [softmax.shape[0], 0.5]))

    # Extract a binary mask of pixels in the rundown phase.
    # maskExtraction(mask,bwconncomp_YesNo,waterValue)
    Ri_down = maskExtractionCV(softmax <= rundown_val, buffer)

    # Extract a binary mask of pixels in the runup phase.
    # maskExtraction(mask,bwconncomp_YesNo,waterValue)
    Ri_up = maskExtractionCV(softmax <= runup_val, buffer)

    # Check if there are valid sample points before interpolation
    if len(xy) < 2:
        print("Insufficient data for interpolation. Skipping interpolation.")
        return np.full(softmax.shape[0], np.nan), Ri_0, Ri_down, Ri_up, rundown_peaks, runup_peaks

    # weight for maxima
    wt1 = scipy.interpolate.pchip_interpolate(xy[:, 0], xy[:, 1], np.arange(len(Ri_0)))
    # weight for minima
    wt2 = -wt1 + 1

    # Compute the weighted average of the runup and rundown masks.
    Ri = np.average([Ri_up, Ri_down], axis=0, weights=[wt1, wt2])
    return Ri, Ri_0, Ri_down, Ri_up, rundown_peaks, runup_peaks

def maskExtractionCV(mask, buffer=[0, 0]):
    """
    Extracts the first occurrence of a nonzero value along each row of a binary mask.
    Optionally applies dilation to buffer the mask before extraction.

    :param mask: (np.ndarray) A 2D binary mask where nonzero values represent the region of interest.
    :param buffer: (list of int, optional) A 2-element list `[buffer_x, buffer_y]` defining the kernel size 
                   for dilation. Default is `[0, 0]`, meaning no buffering is applied.
    :return: (np.ndarray) A 1D array of indices representing the first occurrence of a nonzero value in each row.

    Notes:
        - If `buffer` is specified, the mask is dilated before extracting indices.
        - Uses OpenCV to find and retain the largest connected component in the mask.
        - If no contours are found, returns an array of NaNs with the same row size as `mask`.
    """
    # first_greater_than(A,dim,val):
    Ri_all = first_greater_than(mask, 1, 0)
    if any(buffer):
        # convert to image type for openCV
        mask = mask.astype(np.uint8)
        # buffering using openCV
        kernel = np.ones((buffer[0], buffer[1]), np.uint8)
        mask_dilation = cv2.dilate(mask, kernel, iterations=1)
        # only take largest blob of connected points using openCV
        (cnts, _) = cv2.findContours(
            mask_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            print("No contours found after dilation. Returning NaN series.")
            return np.full(mask.shape[0], np.nan)
        c = max(cnts, key=cv2.contourArea)
        mask_dilation = np.zeros(mask.shape)
        mask_dilation = cv2.fillPoly(mask_dilation, pts=[c], color=(255, 255, 255))
        mask_bw = np.multiply(mask, mask_dilation) > 0
    else:
        # convert to image type for openCV
        mask = mask.astype(np.uint8)
        # buffering using openCV
        # only take largest blob of connected points using openCV
        (cnts, _) = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            print("No contours found. Returning NaN series.")
            return np.full(mask.shape[0], np.nan)
        c = max(cnts, key=cv2.contourArea)
        mask_bw = np.zeros(mask.shape)
        mask_bw = cv2.fillPoly(mask_bw, pts=[c], color=(255, 255, 255)) > 0

    id = np.sum(mask_bw, 1)
    if any(id == 0):
        # fill any empty rows with the original mask
        mask_bw[id == 0, :] = mask[id == 0, :]

    Ri = first_greater_than(mask_bw, 1, 0)
    (nans, _) = nan_helper(Ri)
    x = np.arange(mask.shape[0])
    id = (~nans) & (Ri < mask_bw.shape[1]) & (Ri != 0)
    Ri = np.interp(x, x[id], Ri[id])
    return Ri

def first_greater_than(A, dim, val):
    """
    Find the first index in the array `A` where values exceed `val`, along the specified dimension `dim`.

    :param A: (np.ndarray) The input array.
    :param dim: (int) The dimension along which to search.
    :param val: (float) The threshold value.
    :return (np.ndarray) An array of indices where the first occurrence of `A > val` is found along `dim`.
    """
    # Returns the first index in the array `A` that is greater than `val`, along the specified dimension `dim`.
    id = np.argmax(A > val, axis=dim)
    return id

def nan_helper(y):
    """
    Helper function to handle NaN values in a 1D numpy array.

    :param y: (np.ndarray) A 1D numpy array with possible NaN values.
    :return: (tuple) 
        - nans (np.ndarray): A boolean array indicating positions of NaNs.
        - index (function): A function that converts logical indices of NaNs to numerical indices.
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def pull_datetimes_inDir(folder_path):
    """
    Extract unique dates from filenames in a given folder, based on POSIX timestamps.

    This function scans the filenames in the specified folder, filters out those containing 'runup',
    extracts POSIX timestamps from filenames, and converts them to a unique set of formatted dates.

    :param folder_path: (str) Path to the folder containing the files.
    :return: (list of str) A sorted list of unique dates in 'YYYY-MM-DD' format.
    """
    # Get all the filenames in the folder
    filenames = os.listdir(folder_path)

    # Filter out filenames containing 'runup'
    filenames = [name for name in filenames if 'runup' not in name]

    # Use a set to store unique dates
    unique_dates = set()

    # Process each filename
    for filename in filenames:
        # Remove 'overlay_' prefix and extract the POSIX timestamp
        parts = os.path.basename(filename).replace("overlay_", "")
        posix_timestamp = re.split(r'[.]', parts)[0]  # Extract timestamp before the first '.'
        
        # Convert the POSIX timestamp to a datetime object
        try:
            dt = datetime.fromtimestamp(int(posix_timestamp), tz=timezone.utc)
            # Add the formatted date to the set
            unique_dates.add(dt.strftime("%Y-%m-%d"))  # Format to 'YYYY-MM-DD'
        except ValueError:
            print(f"Skipping invalid timestamp: {posix_timestamp} in file {filename}")

    # Convert the set back to a sorted list (if needed)
    unique_dates = sorted(unique_dates)
    return unique_dates

def plot_multiple_runup(timestackDir, runupDirs):
    """
    Overlays multiple runup lines from different sources onto timestack images and saves the resulting images.

    :param timestackDir: (str) Path to the folder containing timestack images.
    :param runupDirs: (dict) A dictionary where keys are folder names containing runup data 
                          and values are legend labels for the plotted lines.

    :return: None

    Example:
        >>> runup_sources = {"method1": "Runup A", "method2": "Runup B", "method3": "Runup C"}
        >>> plot_multiple_runup("timestacks", runup_sources)

    Notes:
        - Supports `.png`, `.jpg`, and `.jpeg` image formats.
        - Searches for corresponding `runup_***.txt` files in the specified `runupDirs`.
        - Limits plotting to a maximum of **three runup lines** per image.
        - Uses red, green, and blue colors for plotting up to three runup datasets.
        - Saves the overlay images in an `overlays` subfolder within `timestackDir`.
        - Skips files that contain only NaN values or are unreadable.
    """

    overlayDir = os.path.join(timestackDir, 'overlays')
    # Get all image filenames (without extension)
    image_files = [f for f in os.listdir(timestackDir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in image_files:
        image_name = Path(image_file).stem  # Remove extension
        runup_files = []

        # Search for corresponding runup files in runup folders
        for runupDir, legend_name in runupDirs.items():
            runup_path = os.path.join(timestackDir, runupDir, f"runup_{image_name}.txt")
            if os.path.exists(runup_path):
                runup_files.append((runup_path, legend_name))  # Store file path and legend name
          
        # Check if at least one runup file exists
        if len(runup_files) > 0:
            print(f"Processing {image_file} with {len(runup_files)} corresponding runup file(s)")

            # Load the image
            image_path = os.path.join(timestackDir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
            U = image.shape[1]

            # Load the runup file data (skip first row) using genfromtxt (keeps NaNs)
            runup_data = []
            legends = []
            for runup_file, legend_name in runup_files[:5]:  # Limit to 5 runup lines
                try:
                    data = np.genfromtxt(runup_file, delimiter=None)  # Auto-detect delimiter
                    
                    # Skip files that are **entirely NaN**
                    if np.all(np.isnan(data)):
                        print(f"Skipping {runup_file} (all values are NaN)")
                        continue  # Don't plot this file

                    h_runup_id = np.round(U - data).astype(float)  # Keep NaNs as float NaN

                    runup_data.append(h_runup_id)
                    legends.append(legend_name)

                except Exception as e:
                    print(f"Error reading {runup_file}: {e}")
                    continue  # Skip the problematic file

            # Plot the image if at least one valid dataset exists
            if runup_data:
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.axis("off")

                # Overlay the data as lines
                colors = ["r", "g", "b", "y", "k"]  # Up to 3 colors
                for i, data in enumerate(runup_data):
                    plt.plot(data, np.arange(1, image.shape[0] + 1), color=colors[i], label=legends[i], linestyle='-')

                plt.legend()

                # Save the overlay plot
                output_path = os.path.join(overlayDir, f"{image_name}_overlay.jpg")
                plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=300)
                plt.close()  # Close the plot to free memory

                print(f"Saved overlay: {output_path}")

            else:
                print(f"No valid data to plot for {image_file}")
        else:
            print(f"No matching runup files found for {image_file}")

## ------------------- Runup stuff -------------------
def CIRNangles2R(azimuth, tilt, roll):
    """
    Computes a 3x3 rotation matrix R to transform world coordinates 
    to camera coordinates using a ZXZ rotation sequence.

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

def dist_uv_to_xyz(intrinsics, extrinsics, Ud, Vd, known_dim, known_val):
    """
    Converts image coordinates (U, V) to world coordinates (X, Y, Z) 
    using camera intrinsics, extrinsics, and the Direct Linear Transformation (DLT) equations.

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

    # Ensure consistent output format
    xyz = np.column_stack((np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z)))

    return xyz

def get_site_settings(file_path = None):
    """
    Load site settings from a JSON file and convert date strings back to datetime objects.
    The JSON file should be named `site_settings.json` and structured as follows:
    
    ```json
    {
        "SITE_ID": {  // Each site (e.g., "CACO03", "SITEB") is a key containing site-specific information
            "siteName": "Full site name",  // Descriptive name of the site
            "shortName": "Short identifier",  // Abbreviated site name
            "directories": {  // File path locations for different types of data
                "timestackDir": "Path to JPG image files",
                "netcdfDir": "Path to NetCDF files",
                "runupDir": "Path to runup analysis data",
                "topoDir": "Path to topographic data"
            },
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

## ------------------- Save data --------------------
def write_netCDF(site, img_path, U, V, transect_date, xyz, Hrunup, Zrunup):
        
    #history = f"{datetime.now(timezone.utc).isoformat()} Using Python version {sys.version}, netCDF4 version {nc.__version__}, NumPy version {np.__version__}"

    # ----------------- Metadata info -----------------
    fileName = Path(img_path).stem
    fileDatetime = datetime.fromtimestamp(int(fileName.split('.')[0]), tz = pytz.utc)
    onlyDate = fileDatetime.strftime('%Y-%m-%d')

    # Handle sample frequency conversion
    if site["sampling"]["collection_unit"].lower() in ['hz', 'hertz']:
        sample_frequency_Hz = site["sampling"]["sample_frequency"]
    elif site["sampling"]["collection_unit"].lower() in ['s', 'sec', 'seconds']:
        sample_frequency_Hz = 1 / site["sampling"]["sample_frequency"]
    else:
        raise ValueError("Invalid collection_unit. Expected 'Hz', 'Hertz', 's', 'sec', or 'seconds'.")

    # Handle sample period length conversion
    if site["sampling"]["sample_period_unit"].lower() in ['s', 'sec', 'seconds']:
        sample_period_length = site["sampling"]["sample_period_length"] / 60  # Convert seconds to minutes
    elif site["sampling"]["sample_period_unit"].lower() in ['m', 'min', 'minutes']:
        sample_period_length = site["sampling"]["sample_period_length"]  # Already in minutes
    else:
        raise ValueError("Invalid sample_period_unit. Expected 's', 'sec', 'seconds', 'm', 'min', or 'minutes'.")

    site_small = Site(site['siteName'], site['shortName'])
    calibDict = createCalibDict_timestacks(site = site_small, yamlDir = site["directories"]["yamlDir"], fileDatetime = fileDatetime, camNum = site["siteInfo"]["camNum"][1], transect = site["siteInfo"]["rNumber"])
    yamlData = eval(calibDict['data'])


    siteLocation = site["siteInfo"]["siteLocation"]
    utmZone = site["siteInfo"]["utmZone"]
    verticalDatum = site['siteInfo']['verticalDatum']
    freqLimits = site['sampling']['freqLimits']
    global_attrs = {
        "name": fileName,
        "conventions": "CF-1.6",
        "institution": "U.S. Geological Survey",
        "source": "Mounted camera image capture",
        "references": site["siteInfo"]["references"],
        "metadata_link": getattr(site["siteInfo"], "metadata_link", ""),
        "title": f"{site['siteInfo']['siteLocation']} {fileDatetime} UTC: timestack image",
        "program": "Coastal-Marine Hazards and Resources",
        "project": "Next Generation Total Water Level and Coastal Change Forecasts",
        "contributors": site["siteInfo"]["contributors"],
        "year": fileDatetime.year,
        "date": onlyDate,
        "site_location": site["siteInfo"]["siteLocation"],
        "description": f"Pixel intensity timestack sampled at {sample_frequency_Hz:.2f} Hz for {sample_period_length:.2f} minutes at {siteLocation}. Collection began at {fileDatetime} UTC.",
        "sample_period_length": f"{sample_period_length:.2f} minutes",
        "data_origin": site["siteInfo"]["dataOrigin"],
        "coord_system": "UTM",
        "utm_zone": site["siteInfo"]["utmZone"],
        "cam_make": site["siteInfo"]["camMake"],
        "cam_model": site["siteInfo"]["camModel"],
        "cam_lens": site["siteInfo"]["camLens"],
        "data_type": "time series",
        "local_timezone": site["siteInfo"]["timezone"],
        "calibration_parameters": "The following nested dictionary contains the camera's intrinsic and extrinsic calibration parameters computed using the Brown distortion model (Brown, Duane C. May 1966)\n" + str(calibDict),
        "verticalDatum": "NAVD88",
        "verticalDatum_description": "North America Vertical Datum of 1988 (NAVD 88)",
        "freqLimits": site["sampling"]["freqLimits"],
        "freqLimits_description": "Frequency limits on sea-swell and infragravity wave band (Hz). [SS upper, SS/IG, IG lower]"
    }

    # ----------------- Image -----------------
    RAW = cv2.imread(img_path)

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


    dims = {
        "T_dim": len(T),
        "X_dim": len(X),
        "TWLstats_dim": TWLstats.get('S', np.array([])).size
    }
    var_attrs = {
        "U": {
            "long_name": "pixel coordinate along the horizontal axis of the image where timestack was sampled",
            "min_value": float(U.min()),
            "max_value": float(U.max()),
            "units": "pixel",
            "description": f"Pixel coordinate along the horizontal axis (cross-shore) of the image where timestack was sampled at {siteLocation}. Obtained from image collection beginning {transect_date}",
        },
        "V": {
            "long_name": "pixel coordinate along the vertical axis of the image where timestack was sampled",
            "min_value": float(V.min()),
            "max_value": float(V.max()),
            "units": "pixel",
            "description": f"Pixel coordinate along the vertical axis (time) of the image where timestack was sampled at {siteLocation}. Obtained from image collection beginning {transect_date}",
        },
        "T": {
           "standard_name": "time",
            "long_name": "datetime",
            "format": "YYYY-MM-DD HH:mm:SS+00:00",
            "time_zone": "UTC",
            "description": "Times that pixels were sampled to create the timestack. The dimension length is the number of samples in the timestack. Each sample has a time value represented as a datetime.",
            "sample_freq": f"{sample_frequency_Hz} Hertz",
            "sample_length_interval": f"{sample_period_length*60} seconds",
            "min_value": T[0].isoformat(),
            "max_value": T[-1].isoformat(),
        },
        "xUTM":{
            "long_name" : f"Universal Transverse Mercator Zone {utmZone} Easting coordinate of cross-shore timestack pixels",
            "units" : 'meters',
            "min_value" : np.around(UTM_E.min(), decimals=3),
            "max_value" : np.around(UTM_E.max(), decimals=3),
            "description" : f'Cross-shore coordinates of data in the timestack pixels projected onto the beach surface at {siteLocation}. Described using UTM Zone {utmZone} Easting in meters.',
        },
        "yUTM":{
            "long_name" : f'Universal Transverse Mercator Zone {utmZone} Northing coordinate of cross-shore timestack pixels',
            "units" : 'meters',
            "min_value" : np.around(UTM_N.min(), decimals=3),
            "max_value" : np.around(UTM_N.max(), decimals=3),
            "description" : f'Alongshore coordinates of data in the timestack pixels projected onto the beach surface at {siteLocation}. Described using UTM Zone {utmZone} Northing in meters.',
        },
        "z":{
           "long_name" : 'elevation',
           "units" : 'meters',
           "description" : f'Elevation (z-value) in {verticalDatum} of timestack pixels projected onto the beach surface at {siteLocation}.',
           "datum" :site["siteInfo"]["verticalDatum_description"],
           "min_value" : np.around(np.nanmin(z), decimals=3),
           "max_value" : np.around(np.nanmax(z), decimals=3)
        },
        "lon": {
            "long_name" : 'Longitude coordinate of cross-shore timestack pixels',
            "units" : 'degrees_east',
            "standard_name" : 'longitude',
            "description" : f'Location of timestack pixels projected onto the beach surface at {siteLocation}. Described using longitude in decimal degrees.',
            "min_value" : np.around(lon.min(), decimals=5),
            "max_value" : np.around(lon.max(), decimals=5)  
        },
        "lat":{
            "long_name" : 'Latitude coordinate of cross-shore timestack pixels',
            "units" : 'degrees_north',
            "standard_name" : 'latitude',
            "description" : f'Location of timestack pixels projected onto the beach surface at {siteLocation}. Described using latitude in decimal degrees.',
            "min_value" : np.around(lat.min(), decimals=5),
            "max_value" : np.around(lat.max(), decimals=5)
        },
        "Hrunup":{
            "long_name": "cross-shore (horizontal) location of wave runup",
            "min_value": float(Hrunup.min()),
            "max_value": float(Hrunup.max()),
            "units": "meters on local grid",
            "description": "X(Hrunup_id).",
        },
        "Zrunup":{
            "long_name": "total water level elevation timeseries",
            "min_value": float(Hrunup.min()),
            "max_value": float(Hrunup.max()),
            "units": "meters on local grid",
            "description": "Z(Hrunup_id).",
        },
        "crs_utm":{
            "grid_mapping_name" : 'transverse_mercator',
            "scale_factor_at_central_meridian" : 0.999600,
            "longitude_of_central_meridian" : -177 + (int(site["siteInfo"]["utmZone"][0:2]) - 1) * 6,
            "latitude_of_projection_origin" : 0.000000,
            "false_easting" : 500000.000000,
            "false_northing" : 0.000000
        },
        "crs_latlon":{
            "grid_mapping_name": 'latitude_longitude'
        }
    }

    if len(RAW.shape)==3: # rgb
        color_attrs={
            "Color":{
                "long_name": 'image pixel color value',
                "color_band":'RGB',
                "description":'8-bit image color values of the timestack. Three dimensions: time, spatial axis, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2. The horizontal axis of the image is the spatial axis. The different crs_ mappings represent the same coordinates in UTM, local, and longitude/latitude.',
                "coordinates": 'time x_utm', 
                "grid_mapping": 'crs_utm crs_wgs84 crs_local'
            }
        }
        color_dims = ["T_dim", "X_dim", "Color_dim"]
    else: # grayscale
        color_attrs={
            "Color":{
                "long_name": 'image pixel color value',
                "color_band":'Grayscale',
                "description":'8-bit image grayscale values of the timestack. Two dimensions: time, spatial axis. Each value represents a tonal color value between black (0) and white (255). The horizontal axis of the image is the spatial axis. The different crs_ mappings represent the same coordinates in UTM, local, and longitude/latitude.',
                "coordinates": 'time x_utm', 
                "grid_mapping": 'crs_utm crs_wgs84 crs_local'
            }
        }
        color_dims = ["T_dim", "X_dim"]
    
    TWL_attrs = {
        "2exceedence_peaksVar":{
            "long_name": "2 percent exceedence value for twl peaks",
            "units": "meters"
        },
        "2exceedence_notpeaksVar":{
            "long_name": "2 percent exceedence value for twl timeseries",
            "units": "meters"
        },
        "meanVar":{
            "long_name": "mean TWL",
            "units": "meters",
            "description" : "offshore mean twl + wave setup"
        },
        "TpeakVar":{
            "long_name": "peak swash period",
            "units": "seconds",
            "description": "runup (or twl) timeseries peak period"
        },
        "TmeanVar":{
            "long_name": "mean swash period",
            "units": "seconds",
            "description": "runup (or twl) timeseries mean period"
        },
        "SsigVar":{
            "long_name": "significant swash",
            "units":"meters"
        },
        "Ssig_SSVar":{
            "long_name": "significant swash in sea-swell (SS) band",
            "units": "meters",
            "description": f"Between {freqLimits[0]} Hz and {freqLimits[1]} Hz."
        },
        "Ssig_IGVar":{
            "long_name": "significant swash in infragravity (IG) band",
            "units": "meters",
            "description": f"Between {freqLimits[1]} Hz and {freqLimits[2]} Hz."
        },
        "SpectrumVar":{
            "long_name": "runup (or twl) spectral density array",
            "units": "meters^2/Hertz"
        },
        "FrequencyVar":{
            "long_name": "runup (or twl) frequency array",
            "units": "Hertz"
        }
    }

    ds = xr.Dataset(
        data_vars = {
            "u": (["X_dim"], np.around(U, decimals=0), var_attrs["U"]),
            "v": (["X_dim"], np.around(V, decimals=0), var_attrs["V"]),
            "t": (["T_dim"], np.array([t.replace(tzinfo=None) for t in T], dtype="datetime64[ns]"), var_attrs["T"]),
            "x_utm": (["X_dim"], np.around(UTM_E, decimals=3), var_attrs["xUTM"]),
            "y_utm": (["X_dim"], np.around(UTM_N, decimals=3), var_attrs["yUTM"]),
            f'z_{verticalDatum}': (["X_dim"], np.around(z, decimals=3), var_attrs["z"]),
            "lat": (["X_dim"], np.around(lat, decimals=5), var_attrs["lat"]),
            "lon": (["X_dim"], np.around(lon, decimals=5), var_attrs["lon"]),
            "h_runup": (["T_dim"], np.around(Hrunup, decimals=3), var_attrs["Hrunup"]),
            "z_runup": (["T_dim"], np.around(Zrunup, decimals=3), var_attrs["Zrunup"]),
            "color": (color_dims, RAW, color_attrs["Color"]),
            "crs_utm":([], 0, var_attrs["crs_utm"]),
            "crs_latlon":([], 0, var_attrs["crs_latlon"]),
            "TWLstats_2exceedence_peaks":([], TWLstats.get('R2', None), TWL_attrs["2exceedence_peaksVar"]),
            "TWLstats_2exceedence_notpeaks":([], TWLstats.get('eta2', None), TWL_attrs["2exceedence_notpeaksVar"]),
            "TWLstats_mean":([], TWLstats.get('setup', None), TWL_attrs["meanVar"]),
            "TWLstats_Tpeak":([], TWLstats.get('Tp', None), TWL_attrs["TpeakVar"]),
            "TWLstats_Tmean":([], TWLstats.get('Ts', None), TWL_attrs["TmeanVar"]),
            "TWLstats_Ssig":([], TWLstats.get('Ss', None), TWL_attrs["SsigVar"]),
            "TWLstats_Ssig_SS":([], TWLstats.get('Ssin', None), TWL_attrs["Ssig_SSVar"]),
            "TWLstats_Ssig_IG":([], TWLstats.get('Ssig', None), TWL_attrs["Ssig_IGVar"]),
            "TWLstats_spectrum":(["TWLstats_dim"], np.around(TWLstats.get('S', np.array([])), decimals=6), TWL_attrs["SpectrumVar"]),
            "TWLstats_frequency":(["TWLstats_dim"], np.around(TWLstats.get('f', np.array([])), decimals=4), TWL_attrs["FrequencyVar"]),
        },
        coords={
            "T_dim": ("T_dim", np.arange(len(T))), 
            "X_dim": ("X_dim", np.arange(len(UTM_E))),
            "Color_dim": ("Color_dim", np.arange(3))
        },
        attrs = global_attrs
    )

    # Save to NetCDF
    os.makedirs(site["directories"]["netcdfDir"], exist_ok=True)
    output_path = os.path.join(site["directories"]["netcdfDir"], fileName + ".nc")
    ds.to_netcdf(output_path)

    print(f"NetCDF file saved: {output_path}")

    return


# Temp - to be used in traverse_datastore
def process_for_shoreline(site, camera, year, month, day, time, image_type, image_metadata):
    """
    Example processing function for shoreline analysis.

    :param site: Site identifier.
    :param camera: Camera identifier.
    :param year: Year of the image.
    :param month: Month of the image.
    :param day: Day of the image.
    :param time: Time of the image.
    :param image_type: Type of the image.
    :param image_metadata: Metadata dictionary for the image.
    """
    img_path = image_metadata['path']
    print(f"Processing shoreline image: {img_path}")

    try:
        workflow = ShorelineWorkflow(
            image_path=img_path,
            image_type=image_type,
            shoreline_datastore=shoreline_datastore,
            make_intermediate_plots=make_intermediate_plots,
        )
        workflow.process()
        shoreline_datastore.save_shoreline_coords_to_file(
            site=site,
            camera=camera,
            year=year,
            month=month,
            day=day,
            time=time,
            image_type=image_type,
            outputDir="shoreline_output",
        )
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
