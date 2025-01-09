"""
Datastore Module
==========================
This module provides datastore classes to store, retrieve, and manage standard ARGUS products 
and computed shoreline results for various sites and cameras. 
It supports hierarchical storage and exporting of data.

Classes:
    - ImageDatastore: Main class for managing ARGUS-style imagery data.
    - ShorelineDatastore: Main class for managing shoreline-related data.
"""

import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from collections import defaultdict
import numpy as np
import cv2
import csv
from datetime import datetime, timezone
from pvlib import solarposition
from math import radians
import matplotlib.pyplot as plt
import shutil

class ImageDatastore:
    """
    A class to manage and process image data stored in a hierarchical folder structure.

    This class handles storing metadata for images in a nested dictionary structure, 
    allows for image selection, loading, and organizing images based on metadata, 
    and provides methods for filtering and plotting images.

    Attributes:
        images (dict): A nested dictionary structure storing image metadata.
        camera_sites (dict): A dictionary mapping sites to their associated cameras.
    """
    def __init__(self, root_folder=None):
        """
        Initializes the ImageDatastore with a root folder containing image files.

        :param root_folder: (str, optional) The path to the root folder. If None, the user will be prompted to select a folder.
        """
        if root_folder is None:
            self.root_folder = self.select_folder()
        else:
            self.root_folder = root_folder
        # Nested dictionary: site -> camera -> year -> month -> day -> time -> image_type -> path
        self.images = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))))
        self.camera_sites = {}  # Dictionary to store camera site information

# ------------- Folder Selection ------------- 
    @staticmethod
    def select_folder():
        """
        Opens a dialog to select a folder and returns the selected path.

        :returns: (str) The path to the selected folder.
        
        :raises RuntimeError: If folder selection fails.
        """
        try:
            Tk().withdraw()
            folder = askdirectory(title="Select Root Data Folder")
            if not folder:
                raise ValueError("No folder selected. Exiting.")
            return folder
        except Exception as e:
            raise RuntimeError(f"Folder selection failed: {e}")

# ------------- Image Loading and Metadata Parsing ------------- 
    def load_images(self):
        """
        Loads images from the root folder and stores metadata in the images attribute.

        This method parses the filenames and stores metadata such as timestamp, site, camera, and image type.

        :raises ValueError: If a file does not conform to the expected format.
        """
        for root, _, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg', '.tiff')):
                    try:
                        # Parse the filename into its components
                        parts = file.split('.')
                        if len(parts) < 9:
                            # Skip files that do not match the expected format
                            continue

                        # Extract relevant metadata
                        timestamp = parts[0]  # Unix timestamp
                        month = parts[2]      # Month
                        day = parts[3][:2]    # Day of the month
                        time = parts[3][3:]   # Time (HH_MM)
                        year = parts[5]       # Year
                        site = parts[6]       # Site CACO#
                        camera = parts[7]     # Camera (e.g., "c1")
                        image_type = parts[8]   # Image type (e.g., "bright", "dark", "timex", "snap")

                        # Construct full image path
                        full_path = os.path.join(root, file)

                        # Create image metadata dictionary
                        image_metadata = {
                            'timestamp': timestamp,
                            'month': month,
                            'day': day,
                            'time': time,
                            'year': year,
                            'site': site,
                            'camera': camera,
                            'image_type': image_type,
                            'path': full_path
                        }

                        # Store image metadata in the nested dictionary
                        self.images[site][camera][year][month][day][time][image_type].append(image_metadata)

                    except Exception as e:
                        print(f"Failed to parse file {file}: {e}")

# ------------- Copy images to folder
    def copy_images_to_folder(self, destination_folder = None, hierarchical = False):
        """
        Copies all images from the datastore to the specified destination folder.

        :param destination_folder: (str, optional) The path to the destination folder where images will be copied. If None, uses the original directory.
        :param hierarchical: (bool) If True, organizes images into subfolders by site, camera, year, month, and image type. If False, all images are copied to a single folder.
        
        :raises ValueError: If the destination folder is not specified or invalid (when not saving to the original folder).
        """
        # If no destination folder is provided, use the image's original folder.
        if not destination_folder:
            destination_folder = None  # Means to use the image's current path

        # Ensure the destination folder exists, if it is specified
        if destination_folder and not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)

        def traverse_and_copy(site, camera, year, month, images_by_type):
            """
            Helper function to copy images based on the current traversal point.

            :param site: The site identifier.
            :type site: str
            :param camera: The camera identifier.
            :type camera: str
            :param year: The year the image was taken.
            :type year: str
            :param month: The month the image was taken.
            :type month: str
            :param images_by_type: A dictionary of image types to their respective image data.
            :type images_by_type: dict
            """
            for image_type, image_list in images_by_type.items():
                for image_data in image_list:
                    try:
                        # Get image source path
                        src_path = image_data['path']

                        if not os.path.exists(src_path):
                            print(f"Warning: Source file {src_path} does not exist. Skipping.")
                            continue

                        # Determine destination path
                        if destination_folder:
                            if hierarchical:
                                # Create hierarchical folder structure
                                folder_hierarchy = os.path.join(destination_folder, site, camera, year, month, image_type)
                                os.makedirs(folder_hierarchy, exist_ok=True)
                                dst_path = os.path.join(folder_hierarchy, os.path.basename(src_path))
                            else:
                                # Single flat folder
                                dst_path = os.path.join(destination_folder, os.path.basename(src_path))
                        else:
                            # If no destination folder is specified, use the original image's folder
                            dst_path = os.path.join(os.path.dirname(src_path), os.path.basename(src_path))

                        # Copy the image
                        shutil.copy(src_path, dst_path)
                        print(f"Copied {src_path} to {dst_path}")

                    except Exception as e:
                        print(f"Failed to copy {src_path}: {e}")

        # Traverse the nested image datastore
        for site, cameras in self.images.items():
            for camera, years in cameras.items():
                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                traverse_and_copy(site, camera, year, month, images_by_type)

 # ------------- Image Listing ------------- 
    def list_images(self, site = None, camera = None, year = None, month = None):
        """
        Lists images optionally filtered by site, camera, year, and month.

        :param site: (str, optional) The site identifier to filter by.
        :param camera: (str, optional) The camera identifier to filter by.
        :param year: (str, optional) The year to filter by.
        :param month: (str, optional) The month to filter by.
        
        :returns: (dict) A filtered list of images.
        """
        filtered_images = self.images
        if site:
            filtered_images = {site: filtered_images.get(site, {})}
        if camera:
            filtered_images = {site: {camera: filtered_images[site].get(camera, {})} for site in filtered_images}
        if year:
            filtered_images = {site: {cam: {year: data.get(year, {})} for cam, data in cams.items()} 
                               for site, cams in filtered_images.items()}
        if month:
            filtered_images = {site: {cam: {year: {month: data.get(month, {})} for year, data in years.items()}
                               for cam, years in cams.items()} for site, cams in filtered_images.items()}

        # Print results
        for site, cameras in filtered_images.items():
            print(f"Site: {site}")
            for cam, years in cameras.items():
                print(f"  Camera: {cam}")
                for yr, months in years.items():
                    print(f"    Year: {yr}")
                    for mnth, days in months.items():
                        print(f"      Month: {mnth}")
                        for day, times in days.items():
                            print(f"        Day: {day}")
                            for time, types in times.items():
                                print(f"          Time: {time}")
                                for image_type, path in types.items():
                                    print(f"            {image_type}: {path}")

# ------------- Image Statistics ------------- 
    def image_stats(self):
        """
        Displays statistics about the images stored in the datastore.

        :returns: None
        """
        total_images = 0
        days_set = set()
        camera_counts = defaultdict(int)
        type_counts = defaultdict(int)
        camera_sites_stats = defaultdict(lambda: {
            'total_images': 0,
            'days_set': set(),
            'camera_counts': defaultdict(int),
            'type_counts': defaultdict(int)
        })

        # Traverse the nested structure to gather statistics
        for site, cameras in self.images.items():
            for camera, years in cameras.items():
                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                for image_type, image_data_list in images_by_type.items():
                                    for image_metadata in image_data_list:
                                        # Ensure image_data is a dictionary (with metadata)
                                        if isinstance(image_metadata, dict):
                                            # Increment total images
                                            total_images += 1
                                            # Add to days set
                                            days_set.add(f"{month}-{day}")
                                            # Count images by camera and type
                                            camera_counts[image_metadata['camera']] += 1
                                            type_counts[image_metadata['image_type']] += 1

                                            # Update per-site stats
                                            camera_sites_stats[site]['total_images'] += 1
                                            camera_sites_stats[site]['days_set'].add(f"{month}-{day}")
                                            camera_sites_stats[site]['camera_counts'][camera] += 1
                                            camera_sites_stats[site]['type_counts'][image_metadata['image_type']] += 1
                                        else:
                                            print(f"Warning: Expected image metadata, got {type(image_metadata)} instead.")

        # Print statistics
        print(f"\nTotal images: {total_images}")
        print(f"Distinct days: {len(days_set)}")
        print("Images by type:")
        for image_type, count in type_counts.items():
            print(f"  {image_type}: {count}")
        print("Images by camera:")
        for camera, count in camera_counts.items():
            print(f"  {camera}: {count}")

        print("\nCamera site stats:")
        for site, stats in camera_sites_stats.items():
            print(f"Site: {site}")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Distinct days: {len(stats['days_set'])}")
            print(f"  Images by camera:")
            for camera, count in stats['camera_counts'].items():
                print(f"    {camera}: {count}")
            print(f"  Images by type:")
            for image_type, count in stats['type_counts'].items():
                print(f"    {image_type}: {count}")

    def get_image_metadata_by_type(self, image_types, site = None, camera = None):
        """
        Extract metadata for all specified image types, optionally filtering by site and camera.

        :param image_types: (list) A list of image types (e.g., ['bright', 'snap']).
        :param site: (str, optional) The site to filter by. If not provided, all sites are considered.
        :param camera: (str, optional) The camera to filter by. If not provided, all cameras are considered.
        
        :returns: (list) A list of metadata dictionaries for the specified image types, site, and camera.
        """
        images_metadata = []

        # Traverse through the nested structure
        for site_key, cameras in self.images.items():
            if site and site_key != site:  # Skip if site doesn't match the filter
                continue
            
            for camera_key, years in cameras.items():
                if camera and camera_key != camera:  # Skip if camera doesn't match the filter
                    continue

                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                for image_type in image_types:
                                    if image_type in images_by_type:
                                        for image_metadata in images_by_type[image_type]:
                                            images_metadata.append(image_metadata)

        return images_metadata

# ------------- Plotting -------------  
    def _validate_year_and_index(self, year, index):
        """
        Validate the year and index provided for retrieving images.

        :param year: (str) The year of the images.
        :param index: (int) The index of the image within the specified year.

        :returns: (dict or None) The image metadata if valid, else None.
        """
        if not self.images:
            print("No images loaded in the datastore.")
            return None

        # Use the first available year if none is provided
        if year is None:
            year = next(iter(self.images))

        # Validate year existence
        if year not in self.images:
            print(f"No images found for the year: {year}")
            return None

        # Validate index bounds
        if not (0 <= index < len(self.images[year])):
            print(f"Invalid index {index}. Year {year} contains {len(self.images[year])} images.")
            return None

        # Return valid image metadata
        return self.images[year][index]

    def _plot_image(self, image_path, title):
        """
        Helper function to plot an image given its path and title.

        :param image_path: (str) Path to the image file.
        :param title: (str) Title to display on the plot.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def plot_image(self, image_metadata):
        """
        Plot a single image using its metadata dictionary.

        :param image_metadata: (dict) A dictionary containing the image's metadata.
        """
        title = (f"Camera: {image_metadata['camera']}, Type: {image_metadata['image_type']}, "
                f"{image_metadata['month']}, {image_metadata['day']}, {image_metadata['time']}")
        self._plot_image(image_metadata['path'], title)

# ------------- Camera Site Info ------------- 
    def load_camera_sites(self, csv_file = 'camera_sites.csv'):
        """
        Load camera sites data (lat, lon, and camera angle) from a CSV file.

        :param csv_file: (str, optional) The path to the CSV file containing camera site data.
        
        :returns: dict: (dict) A dictionary with site names as keys and tuples (lat, lon, camera_angle) as values.
        
        :raises ValueError: If the data in the CSV file is invalid or incomplete.
        """

        try:
            with open(csv_file, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    site = row['site']
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    angle = float(row['camera_angle'])  # Camera angle (in degrees)
                    self.camera_sites[site] = {'lat': lat, 'lon': lon, 'angle': angle}
            print(f"Loaded camera site info from {csv_file}.")
        except Exception as e:
            print(f"Error loading camera site info from {csv_file}: {e}")

# ------------- Filters -------------

    def filter_black_images(self, threshold = 50, image_type = None):
        """
        Remove images from the datastore that are too dark.

        :param threshold: (int) The brightness threshold below which images are considered "too dark".
        :param image_type: (str, optional) If specified, only filter images of this type (e.g., 'bright', 'snap').

        :return: (list) A list of metadata for the removed images.
        """
        removed_images = []

        for site, cameras in self.images.items():
            for camera, years in cameras.items():
                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                for image_type, image_list in images_by_type.items():
                                    if image_type and image_type != image_type:
                                        continue  # Skip if the image type doesn't match

                                    filtered_images = []
                                    for image_metadata in image_list:
                                        image_path = image_metadata['path']
                                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                                        if image is None:
                                            print(f"Warning: Could not read image {image_path}. Skipping.")
                                            continue

                                        # Convert to grayscale and calculate the mean brightness
                                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                        mean_brightness = np.mean(gray_image)

                                        if mean_brightness >= threshold:
                                            filtered_images.append(image_metadata)
                                        else:
                                            removed_images.append(image_metadata)

                                    # Update the dictionary with the filtered images
                                    self.images[site][camera][year][month][day][time][image_type] = filtered_images

        print(f"Filtered {len(removed_images)} dark images.")
        return removed_images

    def filter_white_images(self, threshold = 200, image_type = None):
        """
        Remove images from the datastore that are too bright.

        :param threshold: (int) The brightness threshold above which images are considered "too bright".
        :param image_type: (str, optional) If specified, only filter images of this type (e.g., 'bright', 'snap').

        :return: (list) A list of metadata for the removed images.
        """
        removed_images = []

        for site, cameras in self.images.items():
            for camera, years in cameras.items():
                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                for image_type, image_list in images_by_type.items():
                                    if image_type and image_type != image_type:
                                        continue  # Skip if the image type doesn't match

                                    filtered_images = []
                                    for image_metadata in image_list:
                                        image_path = image_metadata['path']
                                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                                        if image is None:
                                            print(f"Warning: Could not read image {image_path}. Skipping.")
                                            continue

                                        # Convert to grayscale and calculate the mean brightness
                                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                        mean_brightness = np.mean(gray_image)

                                        if mean_brightness <= threshold:
                                            filtered_images.append(image_metadata)
                                        else:
                                            removed_images.append(image_metadata)

                                    # Update the dictionary with the filtered images
                                    self.images[site][camera][year][month][day][time][image_type] = filtered_images

        print(f"Filtered {len(removed_images)} bright images.")
        return removed_images

    def filter_sun_glare(self, threshold = 30, image_type = None, csv_file = 'camera_sites.csv'):
        """
        Check for potential sun glare in images.

        :param threshold: (int) The angle in degrees above which the image will be considered to have sun glare.
        :param image_type: (str, optional) If specified, only check images of this type (e.g., 'bright', 'snap').
        :param csv_file: (str) The path to the CSV file containing camera site data.

        :return: (list) A list of metadata for the removed images.
        """
        removed_images = []

        # Load camera site information (latitude, longitude, camera angle)
        self.load_camera_sites(csv_file)

        for site, cameras in self.images.items():
            for camera, years in cameras.items():
                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                for image_type, image_list in images_by_type.items():
                                    if image_type and image_type != image_type:
                                        continue  # Skip if the image type doesn't match

                                    filtered_images = []
                                    for image_metadata in image_list:
                                        # Get camera site details
                                        site_data = self.camera_sites.get(site)
                                        if not site_data:
                                            print(f"Warning: No camera site data for {site}. Skipping.")
                                            continue

                                        lat = float(site_data['lat'])
                                        lon = float(site_data['lon'])
                                        camera_angle = float(site_data['angle'])

                                        # Get the Unix timestamp of the image
                                        timestamp = int(image_metadata['timestamp'])
                                        image_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)

                                        # Calculate the sun's position using pvlib
                                        sun_position = solarposition.get_solarposition(image_time, lat, lon)
                                        sun_elevation = sun_position['elevation'].iloc[0]
                                        sun_azimuth = sun_position['azimuth'].iloc[0]

                                        # Check if the sun's elevation is above the threshold for glare
                                        if sun_elevation >= threshold:
                                            sun_azimuth_deg = radians(sun_azimuth)
                                            camera_angle_deg = radians(camera_angle)
                                            azimuth_diff = abs(sun_azimuth_deg - camera_angle_deg)

                                            # If the difference is small (within glare range), flag it as having glare
                                            if azimuth_diff < np.radians(15):
                                                print(f"Sun glare detected in image: {image_metadata['path']}")
                                                removed_images.append(image_metadata)
                                            else:
                                                filtered_images.append(image_metadata)
                                        else:
                                            filtered_images.append(image_metadata)

                                    # Update the dictionary with the filtered images
                                    self.images[site][camera][year][month][day][time][image_type] = filtered_images

        print(f"Checked for sun glare in {len(removed_images)} images.")
        return removed_images

    def filter_blurry_images(self, blur_thresholds = None, image_type = None):
        """
        Remove blurry images from the datastore.

        :param blur_thresholds: (dict, optional) A dictionary where keys are image types 
                                 (e.g., 'bright', 'snap') and values are the Laplacian variance thresholds below 
                                 which images are considered blurry. If not provided, the default threshold will be used.
        :param image_type: (str, optional) If specified, only filter images of this type (e.g., 'bright', 'snap').

        :return: (list) A list of metadata for the removed images.
        """
        removed_images = []
         # Set a default threshold if no thresholds are provided
        default_threshold = 120

        for site, cameras in self.images.items():
            for camera, years in cameras.items():
                for year, months in years.items():
                    for month, days in months.items():
                        for day, times in days.items():
                            for time, images_by_type in times.items():
                                for image_type, image_list in images_by_type.items():
                                    if image_type and image_type != image_type:
                                        continue  # Skip if the image type doesn't match
                                        
                                    # Get the threshold for the image type, default to `default_threshold`
                                    blur_threshold = blur_thresholds.get(image_type, default_threshold) if blur_thresholds else default_threshold

                                    filtered_images = []
                                    for image_metadata in image_list:
                                        image_path = image_metadata['path']
                                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                                        if image is None:
                                            print(f"Warning: Could not read image {image_path}. Skipping.")
                                            continue

                                        # Calculate the Laplacian variance to check blurriness
                                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

                                        if laplacian_var >= blur_threshold:
                                            filtered_images.append(image_metadata)
                                        else:
                                            removed_images.append(image_metadata)

                                    # Update the dictionary with the filtered images
                                    self.images[site][camera][year][month][day][time][image_type] = filtered_images

        print(f"Filtered {len(removed_images)} blurry images.")
        return removed_images


      # c_ShorelineDatastore.py


class ShorelineDatastore:
    """
    A class to manage shoreline data using a nested dictionary structure.

    This class supports hierarchical storage, retrieval, and export of shoreline
    results including error metrics.

    Attributes:
        results (defaultdict): A nested dictionary to store shoreline-related data.
    """
    def __init__(self):
        """
        Initialize the datastore with the root folder containing the images.

        :param root_folder: (str) The path to the root folder containing image files.
                             If None, a dialog will prompt the user to select a folder.
        """
        # Nested dictionary to store results: site -> camera -> year -> month -> day -> time -> image_type -> data_type -> data
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))))

    def store_shoreline_results(self, site, camera, year, month, day, time, image_type, shoreline_coords, bottom_boundary, watershed_coords, y_distance, rmse_value):
        """
        Store the computed results for a given image.

        :param site: (str) The site identifier.
        :param camera: (str) The camera identifier.
        :param year: (str) The year of the image.
        :param month: (str) The month of the image.
        :param day: (str) The day of the image.
        :param time: (str) The time of the image.
        :param image_type: (str) The type of the image (e.g., 'bright').
        :param shoreline_coords: (np.ndarray) The final shoreline coords.
        :param bottom_boundary: (np.ndarray) The computed bottom boundarys. (1,2,3,median)
        :param watershed_coords: (np.ndarray) The watershed coordinates.
        :param y_distance: (np.ndarray) The computed y-distance.
        :param rmse_value: (float) The computed RMSE value.
        """

        # Initialize the data type dictionaries if not present
        if 'shoreline_coords' not in self.results[site][camera][year][month][day][time][image_type]:
            self.results[site][camera][year][month][day][time][image_type]['shoreline_coords'] = []
        if 'bottom_boundary' not in self.results[site][camera][year][month][day][time][image_type]:
            self.results[site][camera][year][month][day][time][image_type]['bottom_boundary'] = []
        if 'watershed_coords' not in self.results[site][camera][year][month][day][time][image_type]:
            self.results[site][camera][year][month][day][time][image_type]['watershed_coords'] = []
        if 'y_distance' not in self.results[site][camera][year][month][day][time][image_type]:
            self.results[site][camera][year][month][day][time][image_type]['y_distance'] = []
        if 'rmse' not in self.results[site][camera][year][month][day][time][image_type]:
            self.results[site][camera][year][month][day][time][image_type]['rmse'] = []

        # Store the results
        self.results[site][camera][year][month][day][time][image_type]['shoreline_coords'].append(shoreline_coords)
        self.results[site][camera][year][month][day][time][image_type]['bottom_boundary'].append(bottom_boundary)
        self.results[site][camera][year][month][day][time][image_type]['watershed_coords'].append(watershed_coords)
        self.results[site][camera][year][month][day][time][image_type]['y_distance'].append(y_distance)
        self.results[site][camera][year][month][day][time][image_type]['rmse'].append(rmse_value)

    def get_shoreline_results(self, site, camera, year, month, day, time, image_type):
        """
        Retrieve the stored results for a given image.

        :param site: (str) The site identifier.
        :param camera: (str) The camera identifier.
        :param year: (str) The year of the image.
        :param month: (str) The month of the image.
        :param day: (str) The day of the image.
        :param time: (str) The time of the image.
        :param image_type: (str) The type of the image (e.g., 'bright').

        :return: (dict) A dictionary containing the shoreline coords, bottom boundary, watershed coords, y-distance, and RMSE values.
        """
        return self.results[site][camera][year][month][day][time][image_type]

    def get_shoreline_coords(self, site, camera, year, month, day, time, image_type):
        """
        Retrieve all filtered bottom boundaries for a given image.

        :param site: (str) The site identifier.
        :param camera: (str) The camera identifier.
        :param year: (str) The year of the image.
        :param month: (str) The month of the image.
        :param day: (str) The day of the image.
        :param time: (str) The time of the image.
        :param image_type: (str) The type of the image (e.g., 'bright').

        :return: (list) A list of all shoreline coords for the given image.
        """
        try:
            # Attempt to retrieve the data from the dictionary
            shoreline_coords = self.results[site][camera][year][month][day][time][image_type]['shoreline_coords']
            return shoreline_coords
        except KeyError:  # If any key is missing, return None
            return None
        except TypeError:  # In case the expected structure is incorrect (e.g., a non-dict value), also return None
            return None
        
    def save_shoreline_coords_to_file(self, site, camera, year, month, day, time, image_type, output_folder = "output"):
        """
        Save shoreline_coords to a text file with a name based on the dictionary keys.

        :param site: (str) The site identifier.
        :param camera: (str) The camera identifier.
        :param year: (str) The year of the image.
        :param month: (str) The month of the image.
        :param day: (str) The day of the image.
        :param time: (str) The time of the image.
        :param image_type: (str) The type of the image (e.g., 'bright').
        :param output_folder: (str) Folder to save the output file. Default is "output".
        """

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok = True)

        # Generate a filename based on the input parameters
        filename = f"{site}_{camera}_{year}_{month}_{day}_{time}_{image_type}_shoreline_coords.txt"
        filepath = os.path.join(output_folder, filename)

        # Retrieve the shoreline coords
        shoreline_coords = self.results[site][camera][year][month][day][time][image_type].get('shoreline_coords', None)

        if shoreline_coords is None:
            print("No shoreline coords found for the specified keys.")
            return

        # Save to the file
        with open(filepath, 'w') as f:
            for coords in shoreline_coords:
                np.savetxt(f, coords, fmt="%.6f", delimiter=",")
                f.write("\n")  # Add a newline between sets of coords for clarity

        print(f"Shoreline coords saved to {filepath}")