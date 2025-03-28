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
# Standard Libraries
import os
import re
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, Counter

# GUI & File Dialogs
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Numerical & Data Processing
import numpy as np
import xarray as xr

# Image Processing
import cv2
import piexif
from PIL import Image
import exiftool

# Geospatial
import utm
import pytz

# Plotting & Visualization
import matplotlib.pyplot as plt

# Custom Utilities
import utils_CIRN
import utils_exif
import utils_segformer
import utils_shoreline
from RunUpTimeseriesFunctions_CHI import *
from segment_anything import SamPredictor, sam_model_registry


class ImageHandler:
    """
    A class to process image data, including reading/writing metadata, rectifying, 
    extracting shorelines, and saving results as NetCDF files.
    """

    def __init__(self, imagePath = None, configPath = None):
        """
        Initializes the ImageHandler with an optional image file.

        :param image_path: (str, optional) Path to the image file. If None, an image must be loaded later.
        """
        self.config = {}
        if configPath:
            if isinstance(configPath, str) and os.path.exists(configPath):
                with open(configPath, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = configPath

        self.image_path = imagePath
        self.image = None  # Placeholder for the image data
        self.image_type = None # Placeholder for image type
        self.image_name = None # Placeholder for image name
        self.datetime = None # Placeholder for datetime
        self.site = None # Placeholder for site name
        self.camera = None # Placehold for camera number
        self.transect = None # Placehold for transect number (if relevant)
        self.metadata = {}  # Dictionary to store metadata
        self.processing_results = {}  # Dictionary to store any processed data

        if imagePath:
            self.load_image(imagePath)
            #self.filter_image()

            if self.image_path:
                try:
                    self.read_metadata()  
                    if not self.metadata:  # Check if metadata extraction failed
                        raise ValueError("Metadata missing")
                except Exception as e:
                    print(f"Metadata missing. Writing metadata...")
                    if all(key in self.config for key in ["jsonDir", "yamlDir"]):
                        self.write_metadata()
                        self.read_metadata()  # Retry after writing metadata
                    else:
                        print("Error: Missing required config values (site, jsonDir, yamlDir).")

    def load_image(self, image_path = None):
        """Loads an image from the given path and extracts basic metadata."""
        image_path = image_path or self.image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.image_path = image_path

        filename = Path(image_path).stem
        # POSIXTIME.DOW.MMM.DD_HH_MM_SS.timezone.YYYY.site.camera.image_type
        filenameElements = filename.split('.')
        epoch_time = filenameElements[0]
        self.image_name = filename
        self.datetime = datetime.fromtimestamp(int(epoch_time), tz=pytz.utc)
        self.site = filenameElements[6]
        self.camera = int(re.findall(r"[-+]?\d*\.\d+|\d+", str(filenameElements[7]))[0])
        self.image_type = filenameElements[8]
        self.transect = int(re.findall(r"[-+]?\d*\.\d+|\d+", str(filenameElements[8]))[0]) if re.findall(r"[-+]?\d*\.\d+|\d+", str(filenameElements[8])) else None

# ------------- Filtering stuff
    def _filter_black_image(self, threshold=50):
        """
        Checks if the image is too dark.

        :param threshold: (int) The brightness threshold below which images are considered "too dark".
        :return: (bool) True if the image is too dark, False otherwise.
        """
        if self.image is None:
            raise ValueError("No image loaded.")

        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(image_cv)

        return mean_brightness < threshold

    def _filter_white_image(self, threshold=200):
        """
        Checks if the image is too bright.

        :param threshold: (int) The brightness threshold above which images are considered "too bright".
        :return: (bool) True if the image is too bright, False otherwise.
        """
        if self.image is None:
            raise ValueError("No image loaded.")

        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(image_cv)

        return mean_brightness > threshold

    def _filter_blurry_image(self, threshold=120):
        """
        Checks if the image is too blurry.

        :param threshold: (int) The Laplacian variance threshold below which images are considered blurry.
        :return: (bool) True if the image is too blurry, False otherwise.
        """
        if self.image is None:
            raise ValueError("No image loaded.")

        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(image_cv, cv2.CV_64F).var()

        return laplacian_var < threshold

    def filter_image(self, black_threshold=50, white_threshold=200, blur_threshold=120):
        """
        Filters the loaded image based on brightness (black/white) and sharpness (blurriness).

        :param black_threshold: (int) Brightness below which images are too dark.
        :param white_threshold: (int) Brightness above which images are too bright.
        :param blur_threshold: (int) Sharpness below which images are too blurry.
        :return: (dict) A dictionary with the filter results.
        """
        if self.image is None:
            raise ValueError("No image loaded.")

        # Default thresholds (fallback)
        default_thresholds = {
            'snap': 20,
            'timex': 15,
            'bright': 35,
            'dark': 20,
            'var': 30
        }
        # Merge provided thresholds with defaults (fallback to defaults if missing keys)
        thresholds = self.config.get("thresholds", None)

        # If thresholds is not a dictionary, fall back to the function input
        if not isinstance(thresholds, dict):
            thresholds = blur_threshold if isinstance(blur_threshold, dict) else {}

        # Merge with defaults (ensure all expected keys exist)
        thresholds = {**default_thresholds, **thresholds}

        is_black = self._filter_black_image(black_threshold)
        is_white = self._filter_white_image(white_threshold)
        is_blurry = self._filter_blurry_image(blur_threshold)

        filtered_out = is_black or is_white or is_blurry

        if filtered_out:
            print('Filtering out')
            # Reset all attributes
            self.image = None
            self.image_path = None
            self.image_type = None
            self.datetime = None
            self.site = None
            self.camera = None
            self.transect = None
            self.metadata = {}
            self.processing_results = {}

        return {
            "is_black": is_black,
            "is_white": is_white,
            "is_blurry": is_blurry,
            "filtered_out": filtered_out,
        }

# -------------Metadata stuff
    def write_metadata(self):
        """
        Processes timestack images by extracting metadata, 
        generating EXIF tags, and writing them back to the images.

        :param site: Site object containing directory paths and calibration information.
        :param jsonDir: (str) Directory containing JSON metadata files.
        :param yamlDir: (str) Directory containing YAML calibration files.
        """

        yamlDir = self.config.get("yamlDir", os.path.join(os.getcwd(), 'YAML'))
        jsonDir = self.config.get("jsonDir", os.path.join(os.getcwd(), 'JSON'))
        # Load JSON Tags

        json_tags = utils_exif.loadMultiCamJson(self.site, self.camera, jsonDir)

        # Prepare EXIF Data
        tags = json_tags['All']
        productTags = json_tags['Products'][self.image_type.split('_')[0]]
        for tag in productTags:
            productTags[tag] = productTags[tag].replace("[insert_creation_datetime]", str(self.datetime) + ' UTC')
        tags.update(productTags)
        
        calibDict = utils_exif.createCalibDict(self.site, camNum = self.camera, yamlDir=yamlDir, fileDatetime=self.datetime, transect=self.transect)
        yamlData = calibDict['data']
        tags.update({
            'GPSDateStamp': yamlData['calibration_date'],
            'GPSTimeStamp': '00:00:00',
            'UserComment': calibDict,
            'PreservedFilename': self.image_name,
            'ImageDescription': tags['Description'],
            'DateTimeOriginal': str(self.datetime)[:19],
            'DateTime': str(datetime.now())
        })
        # Write EXIF Data
        with exiftool.ExifToolHelper(executable='exiftool') as et:
            et.set_tags([self.image_path], tags=tags, params=["-P", "-overwrite_original"])
            keywordList = tags['Keywords'].split(', ')
            for k, keyword in enumerate(keywordList):
                et.execute(f'-keywords{"+" if k else ""}={keyword}', self.image_path, "-P", "-overwrite_original")

    def read_metadata(self):
        image_data = Image.open(self.image_path)
        exif_data = piexif.load(image_data.info.get("exif", b""))
        user_comment_raw = exif_data.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
    
        if not user_comment_raw:
            print(f"No exif data was available for {self.image_path}")
            return None
        
        user_comment = user_comment_raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        if user_comment.startswith("ASCII") or user_comment.startswith("UTF-8"):
            user_comment = user_comment.split(" ", 1)[-1].strip()
    
        first_brace = user_comment.find("{")
        last_brace = user_comment.rfind("}")
        if first_brace != -1 and last_brace != -1:
            user_comment = user_comment[first_brace - len('"data": "') : last_brace]
        
        user_comment_json = user_comment.replace("'", '"').replace('"{', '{').replace('}"', '}')
        user_comment_json = user_comment_json.replace("\\", "").replace('"geo" or "xyz"', "geo or xyz")
        user_comment_json = '{' + user_comment_json + '}'
    
        user_comment_dict = json.loads(user_comment_json)
        
        if "data" in user_comment_dict and isinstance(user_comment_dict["data"], str):
            user_comment_dict["data"] = json.loads(user_comment_dict["data"])
        data_dict = user_comment_dict.get("data", {})
        for key in data_dict:
            try:
                data_dict[key] = float(data_dict[key]) if "." in data_dict[key] else int(data_dict[key])
            except ValueError:
                pass
        extrinsics = {key: data_dict[key] for key in ["x", "y", "z", "azimuth", "tilt", "roll"] if key in data_dict}
        intrinsics = {key: data_dict[key] for key in ["NU", "NV", "coU", "coV", "fx", "fy", "d1", "d2", "d3", "t1", "t2"] if key in data_dict}

        if "transect" in user_comment_dict and isinstance(user_comment_dict["transect"], str):
            user_comment_dict["transect"] = json.loads(user_comment_dict["transect"])
        transect_dict = user_comment_dict.get("transect", {})
        U = np.array(list(map(int, transect_dict.get("U", "").split(","))) if "U" in transect_dict else [])
        V = np.array(list(map(int, transect_dict.get("V", "").split(","))) if "V" in transect_dict else [])
        transect_date = transect_dict.get("transect_date")
        self.metadata = {"extrinsics": extrinsics, "intrinsics": intrinsics, "transect": {"U": U, "V": V, "transect_date": transect_date}}
        
# ------------- Products stuff
    def rectify_image(self):

        if self.image_type in {"snap", "timex", "bright", "dark", "var"}:
            print(f"Rectifying image: {self.image_name}")
            productsPath = self.config.get("productsPath", {}) or utils_CIRN.prompt_for_directory("Select the products JSON file")
            if not productsPath.endswith('.json'):
                productsPath = os.path.join(productsPath, "products.json")  # Append 'products.json' if it's a directory

            with open(productsPath, "r") as file:
                products = json.load(file)
            if isinstance(products, list):
                products_grid = next((item for item in products if item.get("type") == "Grid"), None)
            # If products is already a dictionary, assume it's the desired item
            elif isinstance(products, dict) and products.get("type") == "Grid":
                products_grid = products
            else:
                products_grid = None  # If neither, return None

            output_grid = utils_CIRN.get_xy_coords(products_grid)
            UV_coords = utils_CIRN.get_uv_coords(output_grid, self.metadata['intrinsics'], self.metadata['extrinsics'])
            output_grid = utils_CIRN.get_pixels(output_grid, UV_coords, self.image)

            if any(key not in products_grid or np.isnan(products_grid.get(key, np.nan)) for key in ['east', 'north', 'zone']):
                easting, northing, zone,_ = utm.from_latlon(products_grid['lat'], products_grid['lon'])
            else: 
                easting, northing, zone = products_grid['east'], products_grid['north'], products_grid['zone']
            Ir = output_grid['transect_0']['Ir']                         
            # Store rectified data inside the class
            self.processing_results['rectified_image'] = {
                "Ir": np.array(Ir).astype(np.uint8),  
                "localX": output_grid['transect_0']['localX'].tolist(),
                "localY": output_grid['transect_0']['localY'].tolist(),
                "Eastings": output_grid['transect_0']['Eastings'].tolist(),
                "Northings": output_grid['transect_0']['Northings'].tolist(),
                "Z": output_grid['transect_0']['Z'].tolist(),
                "Origin_Easting": easting,
                "Origin_Northing": northing,
                "Origin_UTMZone": zone,
                "Origin_Angle": products_grid["angle"],
                "dx": products_grid["dx"],
                "dy": products_grid["dy"],
                "tide": products_grid["tide"]
            }
        else:
            print(f"Image was not of correct type (snap, timex, bright, dark, var): {self.image_type}")

# ------------- Timestack stuff
    def run_segformer_on_timestack(self):       
        
        if self.transect:
            # Segformer format - flipped from ARGUS and grayscale
            grayscale_image = cv2.cvtColor(cv2.flip(self.image, 1), cv2.COLOR_RGB2GRAY)
            grayscaleDir = self.config.get("grayscaleDir", os.path.join(os.getcwd(), "grayscale"))
            os.makedirs(grayscaleDir, exist_ok=True)  # Create directory if it doesn't exist
            [os.remove(os.path.join(grayscaleDir, file)) for file in os.listdir(grayscaleDir) if os.path.isfile(os.path.join(grayscaleDir, file)) and file.endswith('flipped_grayscale.png')]
            output_file = os.path.join(grayscaleDir, f"{self.image_name}.flipped_grayscale.png")
            cv2.imwrite(output_file, grayscale_image)
        
            # Get SegFormer weights and model
            weightDir = self.config.get("segformerWeightsDir") if self.config else askdirectory(title="Select folder for SegFormer weights.")
            model = self.config.get("segformerModel", 'SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5')
            weights = os.path.join(weightDir, model)

            if not os.path.exists(weights):
                raise FileNotFoundError(f"SegFormer weights not found in {weightDir}")

            # Run SegFormer model
            segformerCodeDir = self.config.get("segformerCodeDir", os.getcwd())
            subprocess.call(['python', os.path.join(segformerCodeDir, 'segformer.py'), grayscaleDir, weights], stderr=subprocess.PIPE, stdout=subprocess.PIPE, )

    def get_runup_from_segformer(self, runup_val = 0.0, rundown_val = -1.5, save_overlay = True):
        """
        Integrates the `extract_runup` function into the class. It runs shoreline runup extraction on the timestacks.

        :param runup_val: (float) Runup threshold (default: 0).
        :param rundown_val: (float) Rundown threshold (default: -1.5).
        :param save_overlay: (bool) Save overlays as image or not.
        """
    
        runup_val = self.config.get("runup_val", runup_val)
        rundown_val = self.config.get("rundown_val", rundown_val)
        grayscaleDir = self.config.get("grayscaleDir", os.path.join(os.getcwd(), "grayscale"))

        try:
            tomatch_files = {Path(f).stem for f in os.listdir(os.path.join(grayscaleDir, 'meta')) if f.endswith('.npz')}
        except:
            print(f'No matching files in {os.path.join(grayscaleDir, "meta")} for {self.image_name}.')
            return None
        if self.image_name+".flipped_grayscale_res" in tomatch_files:
            npz_data = np.load(os.path.join(grayscaleDir, 'meta', self.image_name +".flipped_grayscale_res.npz"))
            av_prob_stack = npz_data["av_prob_stack"]
            # Runup extraction using softmax-like segmentation image
            try:
                Ri, _, _, _, _, _ = utils_segformer.get_SegGym_runup_pixel_timeseries(av_prob_stack, rundown_val, runup_val, buffer = [0,2])

                I = self.image.astype(np.uint8)
                height, width = I.shape[:2]
                Ri = width - Ri

                outputDir = self.config.get("runupDir", os.path.join(os.getcwd(), "runup"))
                os.makedirs(outputDir, exist_ok=True)  # Create directory if it doesn't exist
                if save_overlay is True:
                    for i in range(1, len(Ri)):
                        # Skip if the current or previous point is NaN
                        if np.isnan(Ri[i-1]) or np.isnan(Ri[i]):
                            continue  # Skip this iteration if either value is NaN
                        x1, y1 = int(Ri[i-1]), i-1
                        x2, y2 = int(Ri[i]), i
                        if 0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height:
                            # Draw a line between consecutive points
                            cv2.line(I, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue line with thickness 2
                    output_path = os.path.join(outputDir, f"overlay_{self.image_name}.jpg")
                    plt.imsave(output_path, I)
                    print(f"Saved overlay image: {output_path}")

                txt_output_path = os.path.join(outputDir, f"runup_{self.image_name}.txt")
                with open(txt_output_path, 'w') as f:
                    for value in Ri:
                        if np.isnan(value):
                            f.write("NaN\n")
                        else:
                            f.write(f"{value}\n")

                self.processing_results['runup'] = {
                    'Ri': Ri,
                    "runup_val": runup_val,
                    "rundown_val": rundown_val
                }
            except:
                print(f'Issue getting runup for {self.image_name}.')

    def compute_runup(self):
        extrinsics, intrinsics, transect_info = self.metadata['extrinsics'], self.metadata['intrinsics'], self.metadata['transect']
        U = np.array(transect_info.get("U", []))
        V = np.array(transect_info.get("V", []))
        transect_date = transect_info.get("transect_date", [])
        
        h_runup_id = np.around(self.processing_results['runup']['Ri']).astype(int)

        # Get DEM and compute shoreline elevation
        DEM = 0#utils_CIRN.get_DEM(date = date, site = site, camNum = camera, transectNum = transectNum)
        xyz = np.zeros((3,3))#utils_CIRN.dist_uv_to_xyz(intrinsics, extrinsics, np.column_stack((U, V)), 'z', DEM['z'])
        Hrunup = U#xyz[h_runup_id, 0:1]
        Zrunup = V#xyz[h_runup_id, 2]
        
        productsPath = self.config.get("productsPath", {}) or utils_CIRN.prompt_for_directory("Select the products JSON file")
        if not productsPath.endswith('.json'):
            productsPath = os.path.join(productsPath, "products.json")  # Append 'products.json' if it's a directory

        with open(productsPath, "r") as file:
            products = json.load(file)
        if isinstance(products, list):
            products_grid = next((item for item in products if item.get("type") == "Grid"), None)
        # If products is already a dictionary, assume it's the desired item
        elif isinstance(products, dict) and products.get("type") == "Grid":
            products_grid = products
        else:
            products_grid = None  # If neither, return None
        easting, northing, zone,_ = utm.from_latlon(products_grid['lat'], products_grid['lon'])
            
        self.processing_results['runup'].update(
            {"U": U,
             "V": V,
             "Origin_Easting": easting,
             "Origin_Northing": northing,
             "Origin_UTMZone": zone,
             "Origin_Angle": products_grid["angle"],
             "DEM": DEM,
             "Eastings": xyz[:,0],
             "Northings": xyz[:,1],
             "Z": xyz[:,2],
             "Hrunup": Hrunup,
             "Zrunup": Zrunup,
             "transect_date_definition": transect_date
        }
        )

# ------------- Shoreline stuff
    def process_bright(self, make_plots = False):
        """
        Process a single 'bright' image.
        
        This method follows the workflow specific to 'bright' image types. It performs the following steps:
        1. Extract bottom boundaries from the image in three attempts (using SAM).
        2. Apply a watershed segmentation to the image based on SAM shoreline.
        3. Compute RMSE between watershed coordinates and SAM shoreline.
        4. Plot and store the results.

        :return: (ShorelineDatastore) The updated shoreline datastore after processing the image.
        """
        print(f"Getting shoreline for bright image: {self.image_name}")

        # Get segment_anything model
        modelDir = self.config.get("segmentAnythingDir") if self.config else askdirectory(title="Select folder for Segment Anything folder.")
        model = self.config.get("segmentAnythingModel", 'sam_vit_h_4b8939.pth')
        model = os.path.join(modelDir, model)

        # Initialize arrays to store bottom boundary estimates from three attempts
        bottom_boundaries = []
        for _ in range(3):
            # Attempt to find surfzone points and predict using SAM model
            coords, _ = utils_shoreline.find_surfzone_coords(self.image_path, num_points = 5, make_plot = make_plots)
            best_mask = utils_shoreline.load_and_predict_sam_model(self.image_path, checkpoint_path = model, shoreline_coords = coords)
            # Extract the bottom boundary from the mask
            bottom_boundary = utils_shoreline.extract_bottom_boundary_from_mask(best_mask, make_plot = make_plots, image_path = self.image_path)
            bottom_boundaries.append(bottom_boundary)

        # Convert bottom boundary attempts into a NumPy array and compute the median bottom boundary
        bottom_boundaries = np.array(bottom_boundaries)  # Shape: (3, n, 2)
        bottom_boundary_median = np.median(bottom_boundaries, axis = 0)  # Shape: (n, 2)

        # Apply watershed segmentation to the image using the median bottom boundary
        _, watershed_coords = utils_shoreline.apply_watershed(self.image_path, bottom_boundary_median, make_plot = make_plots)

        # Clean the boundary coords by removing out-of-bounds values based on mask coordinates
        # mask_x = np.any(best_mask, axis=0)  # Check columns (x-values) for non-zero values
        # x_min = np.argmax(mask_x)  # First non-zero index
        # x_max = len(mask_x) - 1 - np.argmax(mask_x[::-1])  # Last non-zero index

        # Remove coords outside of the x-min to x-max range
        # bottom_boundary_median[(bottom_boundary_median[:, 0] < x_min) | (bottom_boundary_median[:, 0] > x_max), 1] = np.nan
        # watershed_coords[(watershed_coords[:, 0] < x_min) | (watershed_coords[:, 0] > x_max), 1] = np.nan

        # Compute the y-distance and RMSE between the watershed coordinates and the bottom boundary
        y_distance = utils_shoreline.compute_y_distance(watershed_coords, bottom_boundary_median)
        rmse_value = utils_shoreline.compute_rmse(watershed_coords, bottom_boundary_median)
        print(f"RMSE: {np.round(rmse_value, 2)} pixels")

        # Clean shoreline coords based on distance threshold
        shoreline_coords = bottom_boundary_median.copy()
        shoreline_coords[y_distance > 30, :] = np.nan  # Set points beyond 30 pixels to NaN

        # Plot the image and overlay the shoreline and watershed coords
        saveDir = self.config.get("shorelineDir",os.path.join(os.getcwd(), 'shoreline'))
        os.makedirs(saveDir, exist_ok=True)
        utils_shoreline.plot_image_and_shoreline(self.image_path, shoreline_coords = shoreline_coords, watershed_coords = watershed_coords, y_distance = y_distance, save_dir = saveDir)
        plt.close()

        self.processing_results['shoreline'] = {
            "U_coords": bottom_boundaries[0][:, 0],
            "shoreline_coords" : shoreline_coords,
            "bottom_boundary": np.vstack([bottom_boundaries[0][:, 1], bottom_boundaries[1][:, 1], bottom_boundaries[2][:, 1], bottom_boundary_median[:, 1]]).T,
            "watershed_coords": watershed_coords,
            "y_distance": y_distance,
            "rmse_value": rmse_value,
            "model": model
        }
        print("Bright image processing complete.")
    
    def process_timex(self, bright_coords = None, make_plots = False):
        """
        Process a single 'timex' image.
        
        This method follows the workflow specific to 'timex' image types. It uses the results from the 'bright' image 
        processing and follows similar steps to the 'bright' workflow.

        :return: (ShorelineDatastore) The updated shoreline datastore after processing the image.
        """
        print(f"Getting shoreline for timex image: {self.image_name}")
        modelDir = self.config.get("segmentAnythingDir") if self.config else askdirectory(title="Select folder for Segment Anything folder.")
        model = self.config.get("segmentAnythingModel", 'sam_vit_h_4b8939.pth')
        model = os.path.join(modelDir, model)

        # Initialize an array to store bottom boundaries
        bottom_boundaries = []
        for _ in range(3):
            # Generate random points above the bright boundary
            if bright_coords is None:
                water_coords = [(1000, 300)]  # Default coords if no bright data is available
            else:
                water_coords = utils_shoreline.generate_random_coords_above_line(bright_coords.get('shoreline_coords',[]), max_range = 200, min_points = 5, min_y_offset = 100, max_y_offset = 350)
                
            # Predict the bottom boundary using SAM model
            best_mask = utils_shoreline.load_and_predict_sam_model(self.image_path, checkpoint_path=model, shoreline_coords = water_coords)
            bottom_boundary = utils_shoreline.extract_bottom_boundary_from_mask(best_mask, make_plot = make_plots, image_path = self.image_path)
            bottom_boundaries.append(bottom_boundary)

        # Convert bottom boundaries into a NumPy array and compute the median bottom boundary
        bottom_boundaries = np.array(bottom_boundaries)
        bottom_boundary_median = np.median(bottom_boundaries, axis=0)

        shoreline_coords = bottom_boundary_median.copy()
        # Apply watershed segmentation using the median bottom boundary
        _, watershed_coords = utils_shoreline.apply_watershed(self.image_path, bottom_boundary_median, make_plot = make_plots)

        # Compute y-distance and RMSE if there are valid watershed coordinates
        if len(watershed_coords) != 0:
            y_distance = utils_shoreline.compute_y_distance(watershed_coords, shoreline_coords)
            rmse_value = utils_shoreline.compute_rmse(watershed_coords, shoreline_coords)
            print(f"RMSE: {np.round(rmse_value, 2)} pixels")

            # Remove coords where y-distance exceeds threshold
            shoreline_coords[y_distance > 10, :] = np.nan
        else:
            rmse_value = np.nan
            y_distance = None

        # Plot and store results
        saveDir = self.config.get("shorelineDir",os.path.join(os.getcwd(), 'shoreline'))
        os.makedirs(saveDir, exist_ok=True)
        if bright_coords is None:
            utils_shoreline.plot_image_and_shoreline(self.image_path, shoreline_coords = shoreline_coords, watershed_coords = watershed_coords, y_distance = y_distance, save_dir = saveDir)
        else:
            utils_shoreline.plot_image_and_shoreline(self.image_path, shoreline_coords = shoreline_coords, watershed_coords = watershed_coords, other_coords = bright_coords.get('shoreline_coords',[]), y_distance = y_distance, save_dir = saveDir)

            
        # Store the processed results in the datastore
        self.processing_results['shoreline'] = {
            "shoreline_coords" : shoreline_coords,
            "bottom_boundary": np.vstack([bottom_boundaries[0][:, 0], bottom_boundaries[0][:, 1], bottom_boundaries[1][:, 1], bottom_boundaries[2][:, 1], bottom_boundary_median[:, 1]]).T,
            "watershed_coords": watershed_coords,
            "y_distance": y_distance,
            "rmse_value": rmse_value,
            "model": model
        }
        print("Timex image processing complete.")

    def rectify_shoreline(self, tide = 0):
        Ud, Vd = self.processing_results['shoreline'].get('shoreline_coords',[])
        # tide?
        shoreline_xyz = utils_CIRN.dist_uv_to_xyz(self.metadata['intrinsics'], self.metadata['extrinsics'], Ud, Vd, 'z', known_val = tide)
        self.processing_results['shoreline'].update(
            {"rectified_shoreline": shoreline_xyz,
             "tide": tide
            }
        )
        return None
# ------------- Saving
    def save_to_netcdf(self):
        """
        Saves all available data from the ImageHandler class to a NetCDF file, 
        including the image, metadata, rectification data, and configuration settings.

        :param output_path: (str) Path to save the NetCDF file.
        """
        if self.image is None:
            raise ValueError("No image loaded.")
    
        # Core metadata
        sitePath = self.config.get("site_settingsPath", {}) or utils_CIRN.prompt_for_directory("Select the site metadata settings JSON file")
        if not sitePath.endswith('.json'):
            sitePath = os.path.join(sitePath, "site_settings.json")  # Append 'site_settings.json' if it's a directory

        with open(sitePath, "r") as file:
            site = json.load(file)
        site = site[self.site]
                
        onlyDate = self.datetime.strftime('%Y-%m-%d')

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
        utmzone = site["siteInfo"]["utmZone"]

        global_attrs = {
                "name": self.image_name,
                "conventions": "CF-1.6",
                "institution": "U.S. Geological Survey",
                "source": "Mounted camera image capture",
                "references": site["siteInfo"]["references"],
                "metadata_link": getattr(site["siteInfo"], "metadata_link", ""),
                "title": f"{site['siteInfo']['siteLocation']} {self.datetime} UTC: CoastCam image",
                "program": "Coastal-Marine Hazards and Resources",
                "project": "Next Generation Total Water Level and Coastal Change Forecasts",
                "contributors": site["siteInfo"]["contributors"],
                "year": self.datetime.year,
                "date": onlyDate,
                "site_location": site["siteInfo"]["siteLocation"],
                "description": f"Coastcam {self.image_type} image sampled at {sample_frequency_Hz:.2f} Hz for {sample_period_length:.2f} minutes at {self.site}. Collection began at {self.datetime} UTC.",
                "sample_period_length": f"{sample_period_length:.2f} minutes",
                "data_origin": site["siteInfo"]["dataOrigin"],
                "coord_system": "UTM",
                "utm_zone": site["siteInfo"]["utmZone"],
                "cam_make": site["siteInfo"]["camMake"],
                "cam_model": site["siteInfo"]["camModel"],
                "cam_lens": site["siteInfo"]["camLens"],
                "data_type": "imagery",
                "local_timezone": site["siteInfo"]["timezone"],
                "verticalDatum":  site['siteInfo']['verticalDatum'],
                "verticalDatum_description": "North America Vertical Datum of 1988 (NAVD 88)",
                "freqLimits": site["sampling"]["freqLimits"],
                "freqLimits_description": "Frequency limits on sea-swell and infragravity wave band (Hz). [SS upper, SS/IG, IG lower]",
            
                "datetime": str(self.datetime),
                "site": self.site,
                "camera": self.camera,
                "image_path": str(self.image_path),
                "image_type": self.image_type,
                "intrinsics": json.dumps(self.metadata["intrinsics"]),
                "intrinsics_description": "NU, NV, coU, coV, fx, fy, d1, d2, d3, t1, t2",
                "extrinsics": json.dumps(self.metadata["extrinsics"]),
                "intrinsics_description": "Eastings, Northings, Elevation, Azimuth, Tilt, Roll (degrees)"
            }
            
        image_arr = np.array(self.image, dtype=np.uint8)
        
        dims = {
            "U_dim": image_arr.shape[1],
            "V_dim": image_arr.shape[0],
            "Color_dim": 3
        }

        products_attrs = {
                    "I":{
                        "long_name": 'image pixel color value',
                        "color_band":'RGB',
                        "description":'8-bit image color values of the image. Three dimensions: pixel rows, pixel columns, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2.',
                        "coordinates": 'U, V'
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
        
        data_vars={
                "I": (["V_dim", "U_dim", "Color_dim"], image_arr, products_attrs["I"]),
                "crs_utm": ([], 0, products_attrs["crs_utm"]),
                "crs_latlon": ([], 0, products_attrs["crs_latlon"])
        }
        
        coords={
                "U_dim": ("U_dim", np.arange(dims["U_dim"])),
                "V_dim": ("V_dim", np.arange(dims["V_dim"])),
                "Color_dim": ("Color_dim", np.arange(dims["Color_dim"]))
        }
            
        rectified_image = self.processing_results.get("rectified_image", {})
        shoreline = self.processing_results.get("shoreline", {})
        runup = self.processing_results.get("runup", {})

        if rectified_image:
            print("Saving rectified image")
            global_attrs.update({"origin_easting": rectified_image.get("Origin_Easting", np.nan),
                                "origin_northing": rectified_image.get("Origin_Northing", np.nan),
                                "origin_UTMZone": rectified_image.get("Origin_UTMZone", np.nan),
                                "origin_angle": rectified_image.get("Origin_Angle", np.nan),
                                "origin_angle_units": "degrees",
                                "dx": rectified_image.get("dx", np.nan),
                                "dy": rectified_image.get("dy", np.nan),
                                "dx_dy_units": "meters",
                                "tide": rectified_image.get("tide", np.nan),
                                "tide_description": f"tide level used in projection, {site['siteInfo']['verticalDatum']}m"
            })
            
            dims.update({"X_dim": np.shape(np.array(rectified_image.get("localX", [])))[0],
                        "Y_dim": np.shape(np.array(rectified_image.get("localY", [])))[1],
            })
            
            products_attrs.update({
                    "Ir":{
                        "long_name": 'rectified image pixel color value',
                        "color_band":'RGB',
                        "description":'Rectified image. Three dimensions: X, Y, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2.',
                        "coordinates": 'X,Y', 
                        "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                        },
                    "localX":{
                        "long_name": "Local cross-shore coordinates in meters of rectified image.",
                        "min_value": np.around(np.array(rectified_image.get("localX", [])).min(), decimals=3),
                        "max_value": np.around(np.array(rectified_image.get("localX", [])).min(), decimals=3),
                        "units": "meters",
                        "description":" Local cross-shore coordinates in meters of rectified image. Rotated based on shorenormal angle and origin. ",
                        },
                    "localY":{
                        "long_name": "Local along-shore coordinates in meters of rectified image.",
                        "min_value": np.around(np.array(rectified_image.get("localY", [])).min(), decimals=3),
                        "max_value": np.around(np.array(rectified_image.get("localY", [])).min(), decimals=3),
                        "units": "meters",
                        "description":" Local along-shore coordinates in meters of rectified image. Rotated based on shorenormal angle and origin. ",
                    },
                    "Eastings":{
                        "long_name" : f"Universal Transverse Mercator Zone {utmzone} Easting coordinate of rectified image",
                        "units" : 'meters',
                        "min_value" : np.around(np.array(rectified_image.get("Eastings", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(rectified_image.get("Eastings", [])).max(), decimals=3),
                        "description" : f'Cross-shore coordinates of data in the rectified image projected onto the beach surface at {self.site}. Described using UTM Zone {site["siteInfo"]["utmZone"]} Easting in meters.',
                    },
                    "Northings":{
                        "long_name" : f"Universal Transverse Mercator Zone {utmzone} Northing coordinate of rectified image",
                        "units" : 'meters',
                        "min_value" : np.around(np.array(rectified_image.get("Northings", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(rectified_image.get("Northings", [])).max(), decimals=3),
                        "description" : f'Cross-shore coordinates of data in the rectified image projected onto the beach surface at {self.site}. Described using UTM Zone {site["siteInfo"]["utmZone"]} Northing in meters.',
                    },
                    "Z":{
                        "long_name" : 'elevation',
                        "units" : 'meters',
                        "coordinates": 'X,Y',
                        "description" : f'Elevation (z-value) in {site["siteInfo"]["verticalDatum"]} of rectified image projected onto the beach surface at {self.site}.',
                        "datum": site["siteInfo"]["verticalDatum_description"],
                        "min_value" : np.around(np.nanmin(np.array(rectified_image.get("Z", []))), decimals=3),
                        "max_value" : np.around(np.nanmax(np.array(rectified_image.get("Z", []))), decimals=3)
                    }
            })

            coords.update({"X_dim": ("X_dim", np.arange(dims["X_dim"])),
                           "Y_dim": ("Y_dim", np.arange(dims["Y_dim"]))
            })

            data_vars.update({"Ir": (["X_dim", "Y_dim", "Color_dim"], np.array(rectified_image.get("Ir", []), dtype=np.uint8), products_attrs["Ir"]),
                            "localX": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("localX", [])), decimals=3), products_attrs["localX"]),
                            "localY": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("localY", [])), decimals=3), products_attrs["localY"]),
                            "Eastings": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("Eastings", [])), decimals=3), products_attrs["Eastings"]),
                            "Northings": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("Northings", [])), decimals=3), products_attrs["Northings"]),
                            "Z": (["X_dim", "Y_dim"], np.around(np.array(rectified_image.get("Z", [])), decimals=3), products_attrs["Z"]),
            })
    
        if shoreline: 
            print("Saving shoreline")
            global_attrs.update({"tide": shoreline.get("tide", 0),
                                "tide_description": f"tide level used in projection, {site['siteInfo']['verticalDatum']}m",
                                "SAM_model": shoreline.get("model",{})
            })
            
            dims.update({"XYZ_dim": 3,
                        "UV_dim": 2,
                        "dim_4": 4
            })
           
            products_attrs.update({
                    "shoreline_rectified_coords":{
                        "long_name": 'UTM coordinates of shoreline from oblique image.',
                        "description":'Rectified shoreline coordinates on UTM using calculated EO and IO. Projected to given tide level.',
                        "coordinates": 'X, Y, Z',
                        "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                    },
                    "shoreline_pixel_coords":{
                        "long_name": '[U,V] coordinates of the shoreline from the oblique image.',
                        "description":'Shoreline as determined by the median of the SAM and large errors with the watershed algorithm are removed.',
                        "coordinates": 'U, V', 
                        "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                        },
                    "bottom_boundary":{
                        "long_name": "Segment anything determined shoreline",
                        "units": "pixels",
                        "description":"First 3 columns: 3 attemps at determining the shoreline from segment anything. 4th column: Vertical median of 3 shorelines. Used to compare with watershed algorithm shoreline and basis for final shoreline. Markers for ocean initialized from either: bright) random points spread across the largest, brightest continous area or timex) points at least 10 pixels above the matching brightest shoreline or point (1000,300).",
                        },
                    "watershed_coords":{
                        "long_name": "[U,V] coordinates of the shoreline as determined by a watershed algorithm.",
                        "units": "pixels",
                        "coordinates": 'U, V',
                        "description":" Markers for ocean initialized from points at least 10 pixels above the median SAM bottom boundary, markers for sand from points at least 10 pixels below.",
                    },
                    "y_distance":{
                        "long_name" : "Vertical distance in pixels between watershed_coords and median bottom boundary (SAM). ",
                        "units" : 'pixels',
                        "min_value" : np.around(np.array(shoreline.get("y_distance", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(shoreline.get("y_distance", [])).max(), decimals=3),
                        "description" :  'Any horizontal locations that have a y_distance > 30 pixels are excluded from the final shoreline. Helps with stability of the estimates. '
                    },
                    "rmse_value":{
                        "long_name" : "Mean rmse of vertical distance in pixels between watershed_coords and median bottom boundary (SAM).",
                        "units" : 'pixels',
                        "min_value" : np.around(np.array(shoreline.get("rmse_value", [])).min(), decimals=3),
                        "max_value" : np.around(np.array(shoreline.get("rmse_value", [])).max(), decimals=3),
                        "description" : "Provides general estimate of error of shoreline prediction.",
                    }
            })

            data_vars.update({"shoreline_rectified_coords": (["U_dim", "XYZ_dim"], np.array(shoreline.get("rectified_shoreline", np.full((dims["U_dim"], 3), np.nan))), products_attrs["shoreline_rectified_coords"]),
                              "shoreline_pixel_coords": (["U_dim", "UV_dim"], np.array(shoreline.get("shoreline_coords",[])), products_attrs["shoreline_pixel_coords"]),
                              "bottom_boundary_attemps": (["U_dim", "dim_4"], np.array(shoreline.get("bottom_boundary",[])), products_attrs["bottom_boundary"]),
                              "watershed_coords": (["U_dim", "UV_dim"], np.array(shoreline.get("watershed_coords",[])), products_attrs["watershed_coords"]),
                              "y_distance": (["U_dim"], np.array(shoreline.get("y_distance",[])), products_attrs["y_distance"]),
                              "rmse_value": ([], np.array(shoreline.get("rmse_value",[])), products_attrs["rmse_value"])
            })  
            
        if runup:
            print("Saving runup")
            TWLstats = runup.get('TWLstats')
            verticalDatum = site['siteInfo']['verticalDatum']
            freqLimits = site['sampling']['freqLimits']

            t_sec = np.around(np.arange(0, sample_period_length*60, 1/sample_frequency_Hz)[:self.image.shape[0]], decimals=3) # timeseries in seconds
            T = np.array([self.datetime + timedelta(seconds = t) for t in t_sec])

            global_attrs.update({"origin_easting": runup.get("Origin_Easting", np.nan),
                                "origin_northing": runup.get("Origin_Northing", np.nan),
                                "origin_UTMZone": runup.get("Origin_UTMZone", np.nan),
                                "origin_angle": runup.get("Origin_Angle", np.nan),
                                "origin_angle_units": "degrees",
                                "tide": runup.get("tide", np.nan),
                                "tide_description": f"tide level used in projection, {site['siteInfo']['verticalDatum']}m",
                                "transect_date_definition": runup.get("transect_date_definition"),
                                "transect_date_description": "date where U,V coordinates were defined based on recent EO."
            })
            
            dims.update({"X_dim": np.shape(self.image)[0],
                        "T_dim": np.shape(self.image)[1],
                        "TWLstats_dim": runup.get('TWLstats').get('S', np.array([])).size
            })

            coords.update({"X_dim": ("X_dim", np.arange(dims["X_dim"])),
                           "T_dim": T
            })

            products_attrs.update({
                                "I":{
                                    "long_name": 'image pixel color value',
                                    "color_band":'RGB',
                                    "description":'8-bit image color values of the timestack. Three dimensions: time, spatial axis, color band, where the colors are RGB--R(ed) color band equals 0, G(reen) color band equals 1, and B(blue) color band equals 2. The horizontal axis of the image is the spatial axis. The different crs_ mappings represent the same coordinates in UTM, local, and longitude/latitude.',
                                    "coordinates": "time x_utm",
                                    "grid_mapping": 'crs_utm crs_wgs84 crs_local'
                                },
                                "Ri":{
                                    "long_name": "pixel coordinate of runup",
                                    "description": "pixel coordinate of wave runup line as found by the segformer model.",
                                    "units": "pixels"
                                },
                                "runup_val":{
                                    "description": "Runup value used to extract runup from softmax score."
                                },
                                "rundown_val":{
                                    "description": "Rundown value used to extract rundown from softmax score."
                                },
                                "U_transect": {
                                    "long_name": "pixel coordinate along the horizontal axis of the image where timestack was sampled",
                                    "min_value": float(np.min(runup.get("U",[]))),
                                    "max_value": float(np.max(runup.get("U",[]))),
                                    "units": "pixel",
                                    "description": f"Pixel coordinate along the horizontal axis (cross-shore) of the image where timestack was sampled at {self.site}. Obtained from image collection beginning {transect_date}",
                                },
                                "V_transect": {
                                    "long_name": "pixel coordinate along the vertical axis of the image where timestack was sampled",
                                    "min_value": float(np.min(runup.get("V",[]))),
                                    "max_value": float(np.max(runup.get("V",[]))),
                                    "units": "pixel",
                                    "description": f"Pixel coordinate along the vertical axis (time) of the image where timestack was sampled at {self.site}. Obtained from image collection beginning {transect_date}",
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
                                "Eastings":{
                                    "long_name" : f"Universal Transverse Mercator Zone {utmZone} Easting coordinate of cross-shore timestack pixels",
                                    "units" : 'meters',
                                    "min_value" : np.around(np.min(runup.get("Eastings",[])), decimals=3),
                                    "max_value" : np.around(np.max(runup.get("Eastings",[])), decimals=3),
                                    "description" : f'Eastings coordinates of data in the timestack pixels projected onto the beach surface at {self.site}. Described using UTM Zone {utmZone} Easting in meters.',
                                },
                                "Northings":{
                                    "long_name" : f'Universal Transverse Mercator Zone {utmZone} Northing coordinate of cross-shore timestack pixels',
                                    "units" : 'meters',
                                    "min_value" : np.around(np.min(runup.get("Northings",[])), decimals=3),
                                    "max_value" : np.around(np.max(runup.get("Northings",[])), decimals=3),
                                    "description" : f'Northings coordinates of data in the timestack pixels projected onto the beach surface at {self.site}. Described using UTM Zone {utmZone} Northing in meters.',
                                },
                                "Z":{
                                    "long_name" : 'Elevation',
                                    "units" : f"{site['siteInfo']['verticalDatum']}m",
                                    "description" : f'Elevation (z-value) in {verticalDatum} of timestack pixels projected onto the beach surface at {self.site}.',
                                    "datum" :site["siteInfo"]["verticalDatum"],
                                    "min_value" : np.around(np.nanmin(runup.get("Z",[])), decimals=3),
                                    "max_value" : np.around(np.nanmax(runup.get("Z",[])), decimals=3)
                                },
                                "DEM":{
                                    "description": "Digital Elevation Model used for runup projection.",
                                    "units": "meters"
                                },
                                "Hrunup":{
                                    "long_name": f"[X, Y] coordinates of wave runup in Eastings, Northings for UTM zone {utmZone}.",
                                    "min_value": [np.min(runup.get('Hrunup',0)[:,0]), np.min(runup.get('Hrunup',0)[:,1])],
                                    "max_value": [np.max(runup.get('Hrunup',0)[:,0]), np.max(runup.get('Hrunup',0)[:,1])],
                                    "units": "meters ",
                                    "description": "Horizontal coordinates of wave runup as converted from U,V coordinates to Eastings, Northings from EO/IO.",
                                    "coordinates": "X,Y"
                                },
                                "TWL":{
                                    "long_name": "Total water level elevation timeseries",
                                    "min_value": float(runup.get('Zrunup',0).min()),
                                    "max_value": float(runup.get('Zrunup',0).max()),
                                    "units": f"{site['siteInfo']['verticalDatum']}m",
                                    "description": "Vertical elevation of wave runup (total water level).",
                                }    
                        })
            
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

            data_vars.update({"I": (["U_dim", "V_dim", "Color_dim"], image_arr, products_attrs["I"]),
                            "Ri": (["T_dim"], np.array(runup.get("Ri",[])), products_attrs["Ri"]),
                            "runup_val": ([], np.array(runup.get("runup_val",[])), products_attrs["runup_val"]),
                            "rundown_val": ([], np.array(runup.get("rundown_val",[])), products_attrs["rundown_val"]),
                            "U_transect": (["X_dim"], np.array(runup.get("U",[])), products_attrs["U_transect"]),
                            "V_transect": (["X_dim"], np.array(runup.get("V",[])), products_attrs["V_transect"]),
                            "Eastings": (["X_dim"], np.array(runup.get("Eastings",[])), products_attrs["Eastings"]),
                            "Northings": (["X_dim"], np.array(runup.get("Northings",[])), products_attrs["Northings"]),
                            "Z": (["X_dim"], np.array(runup.get("Z",[])), products_attrs["Z"]),
                            "T": (["T_dim"], np.array(T), products_attrs["T"]),
                            "DEM": ([], np.array(runup.get("DEM",[])), products_attrs["DEM"]),
                            "Hrunup": (["T_dim", 2], np.array(runup.get("Hrunup",[])), products_attrs["Hrunup"]),
                            "TWL": (["T_dim"], np.array(runup.get("Zrunup",[])), products_attrs["TWL"]),
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
            })  
        
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = coords,
            attrs = global_attrs
        )
      
        # Save to NetCDF
        saveDir = self.config.get("netcdfDir", os.path.join(os.getcwd(), 'netcdf'))
        os.makedirs(saveDir, exist_ok = True)
        saveName = os.path.join(saveDir, self.image_name + ".nc")
        ds.to_netcdf(saveName)
        print(f"All data saved to {saveName}")


class ImageDatastore:
    """
    A class to manage and process image data stored in a hierarchical folder structure.

    This class handles storing metadata for images in a nested dictionary structure, 
    allows for image selection, loading, and organizing images based on metadata, 
    and provides methods for filtering and plotting images.

    Attributes:
        images (list): A list of images in the folder
    """
    def __init__(self, configPath = None, imageDir = None):
        """
        Initializes the ImageDatastore with a root folder containing image files.

        :param rootDir: (str, optional) The path to the root folder. If None, the user will be prompted to select a folder.
        :param configPath: (str, optional) The path to the config.json file.
        """
        
        self.config = {}
        if configPath:
            if isinstance(configPath, str) and os.path.exists(configPath):
                with open(configPath, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = configPath
        self.imageDir = self.config.get('imageDir', imageDir) or self.selectDir()
        self.images = {}

        self.load_images()
        self.images = {path: data for path, data in self.images.items() if '.ras.' not in path}


# ------------- Folder Selection ------------- 
    @staticmethod
    def selectDir():
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
    def load_images(self, top_folder_only=False):
        for root, _, files in os.walk(self.imageDir, topdown=True):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg', '.tiff')):
                    full_path = os.path.join(root, file)
                    self.images[full_path] = None
            if top_folder_only:
                break

    def initialize_image_handlers(self):
        """Initialize ImageHandler instances for all stored images."""
        for image_path in self.images.keys():
            self.images[image_path] = ImageHandler(imagePath = image_path, configPath = self.config)

    def apply_to_all_images(self, func_name, *args, **kwargs):
        """Apply a function from ImageHandler to all images."""
        for image_path, image_handler in self.images.items():
            if image_handler is None:
                image_handler = ImageHandler(image_path, config = self.config)
                self.images[image_path] = image_handler  # Store the instance

            if hasattr(image_handler, func_name):
                getattr(image_handler, func_name)(*args, **kwargs)
            else:
                print(f"Function '{func_name}' not found in ImageHandler.")

# ------------- Filtering options
    def filter_unmatched_images(self, matchDir):
        """
        Removes images from the datastore that do not have a corresponding file in the matchfolder.

        :param matchDir: (str) Folder containing the corresponding files.
        """
        tomatch_files = {Path(f).stem for f in os.listdir(matchDir)}
        self.images = {img: handler for img, handler in self.images.items() if Path(img).stem in tomatch_files}

    def get_unique_image_types(self):
        """
        Retrieves all unique image types from the datastore.

        :return: (set) A set of unique image types.
        """
        return {Path(img).stem.split('.')[-1] for img in self.images}

    def keep_images_by_type(self, image_types_to_keep):
        """
        Keeps only images of a specified type in the datastore.

        :param image_types_to_keep: (str) The type of images to keep (e.g., 'transect', 'bright').
        """
        self.images = {
            img: handler for img, handler in self.images.items()
            if any(image_type in img for image_type in image_types_to_keep)
        }
    
    def remove_images_by_type(self, image_types_to_remove):
        """
        Removes images of specified types from the datastore.

        :param image_types_to_remove: (list or str) The type(s) of images to remove (e.g., 'transect', 'bright').
        """
        if isinstance(image_types_to_remove, str):  # If a single string is passed, convert it to a list
            image_types_to_remove = [image_types_to_remove]
        
        self.images = {
            img: handler for img, handler in self.images.items()
            if not any(image_type in img for image_type in image_types_to_remove)
        }

    def get_image_metadata_by_type(self, image_types, site=None, camera=None):
        """
        Retrieves metadata for images filtered by type, site, and camera.

        :param image_types: (list) List of image types (e.g., ['timex', 'snap']).
        :param site: (str, optional) Site identifier to filter images.
        :param camera: (str, optional) Camera identifier to filter images.
        :return: (list) List of dictionaries containing image metadata.
        """
        image_metadata = []

        image_types_flat = image_types[0] if image_types and isinstance(image_types, list) and isinstance(image_types[0], list) else image_types

        for img_path in self.images.keys():
            metadata = {
                "path": img_path,
                "name": self.images[img_path].image_name,
                "site": self.images[img_path].site,  
                "camera": self.images[img_path].camera, 
                "image_type": self.images[img_path].image_type,
                "timestamp": self.images[img_path].datetime
            }
            # Apply filtering
            if (not site or metadata["site"] == site) and \
                (not camera or int(metadata["camera"]) == int(camera)) and \
               (not image_types_flat or metadata["image_type"] in image_types_flat):
                image_metadata.append(metadata) # and \

        return image_metadata

    def image_stats(self):
        """
        Displays statistics about the images stored in the datastore.

        :returns: None
        """
        total_images = len(self.images)
        type_counts = Counter(Path(img).stem.split('.')[-1] for img in self.images)
        print(f"Total images: {total_images}")
        print("Image types and their counts:")
        for image_type, count in type_counts.items():
            print(f"  {image_type}: {count}")

    def list_images(self):
        """
        Lists images optionally filtered by site, camera, year, and month.

        :param site: (str, optional) The site identifier to filter by.
        :param camera: (str, optional) The camera identifier to filter by.
        :param year: (str, optional) The year to filter by.
        :param month: (str, optional) The month to filter by.
        
        :returns: (dict) A filtered list of images.
        """
        filtered_images = self.images.keys()
        for img in filtered_images:
            print(img)

# ------------- Products stuff
    def merge_images(self):
        """
        Merges images from multiple rectified images generated on-the-fly.

        Steps:
        1. Groups images from self.images that belong to the same scene (ignoring camera number).
        2. Runs `rectify_image` on each matching image.
        3. Merges the rectified images into a single composite.
        4. Saves and displays the merged rectified image.

        Parameters:
        - merged_rectifiedDir (str, optional): Directory to save output images.
        - products (dict, optional): Dictionary containing lat/lon coordinates for overlay.

        Returns:
        - dict: A dictionary containing merged images and associated geographic data.
        """
        
        if not self.images:
            print("No images loaded in datastore.")
            return {}
        
        merged_rectifiedDir = self.config.get("merged_rectifiedDir",os.path.join(os.getcwd(), 'merged_images'))
        os.makedirs(merged_rectifiedDir, exist_ok=True)
        productsPath = self.config.get("productsPath", {})
        if productsPath:
            if not productsPath.endswith('.json'):
                productsPath = os.path.join(productsPath, "products.json")  # Append 'products.json' if it's a directory
            try:
                with open(productsPath, "r") as file:
                    products = json.load(file)
                if isinstance(products, list):
                    products_grid = next((item for item in products if item.get("type") == "Grid"), None)
                # If products is already a dictionary, assume it's the desired item
                elif isinstance(products, dict) and products.get("type") == "Grid":
                    products_grid = products
                else:
                    products_grid = None  # If neither, return None
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load products file ({productsPath}): {e}")
                products_grid = None

        # Group images by removing the camera number (assume format includes `.cX.` where X is a digit)
        grouped_images = defaultdict(list)
        for img_path in self.images.keys():
            base_name = re.sub(r"\.c\d+", "", Path(img_path).stem)  # Remove ".cX" to group by scene
            grouped_images[base_name].append(img_path)

        self.rectifications = {}

        for base_name, image_paths in grouped_images.items():
            print(f"Processing: {base_name} with {len(image_paths)} images")
            
            images = []
            eastings, northings, localX, localY = None, None, None, None  # Initialize once

            for img_path in image_paths:
                # Ensure we are using an ImageHandler instance
                if isinstance(self.images, list) or not self.images[img_path]:  # If self.images is a list of file paths
                    self.images[img_path] = ImageHandler(imagePath = img_path, configPath = self.config)
                
                img_handler = self.images[img_path] 
                
                # Run rectification if not already done
                if not img_handler.processing_results and "rectified_image" not in img_handler.processing_results:
                    print(f'Running rectify_image on {img_path}.')
                    img_handler.rectify_image()

                if "rectified_image" in img_handler.processing_results:
                    data = img_handler.processing_results["rectified_image"]
                    
                    I = np.array(data.get("Ir"))
                    I = np.expand_dims(I, axis=-1) if I.ndim == 2 else I  # Ensure shape (x, y, 1)
                    
                    eastings = np.array(data.get("Eastings", eastings))
                    northings = np.array(data.get("Northings", northings))
                    localX = np.array(data.get("localX", localX))
                    localY = np.array(data.get("localY", localY))

                    images.append(I)
                else:
                    print(f"Warning: No rectification data for {img_path}")

            if images:
                images = np.stack(images, axis=-1)  # Shape (x, y, 3, num_cameras)
                Ir = utils_CIRN.camera_seam_blend(images)  # Merge rectified images

                self.rectifications[base_name] = {
                    "Ir": Ir,
                    "Eastings": eastings,
                    "Northings": northings,
                    "localX": localX,
                    "localY": localY
                }

                # Create visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f"Rectified Image - {base_name}", fontsize=14)

                # First subplot: pcolor
                ax1 = axes[0]
                pcolor_plot = ax1.pcolor(eastings, northings, Ir.squeeze(), shading='auto')
                if products_grid:
                    easting, northing, _, _ = utm.from_latlon(products_grid['lat'], products_grid['lon'])
                    ax1.scatter(easting, northing, c='r')
                ax1.set_xlabel("Eastings")
                ax1.set_ylabel("Northings")

                # Second subplot: imshow
                ax2 = axes[1]
                im = ax2.imshow(
                    Ir.squeeze(),
                    extent=[localX[0, 0], localX[-1, -1], localY[0, 0], localY[-1, -1]],
                    origin='lower'
                )
                ax2.set_xlabel("Local X")
                ax2.set_ylabel("Local Y")

                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title

                if merged_rectifiedDir:
                    os.makedirs(merged_rectifiedDir, exist_ok=True)
                    save_path = os.path.join(merged_rectifiedDir, f"{base_name}.merged_rectified.png")
                    plt.savefig(save_path, dpi=300)
                    print(f"Saved merged rectified image to {save_path}")

                plt.show()

    def create_video(self, image_type = 'timex', frame_rate = 24, video_name = None, videoDir = os.getcwd(), camera = None, site = None, start_time = None, end_time = None):
        """
        Create a video from images in the datastore with optional filtering by image type, camera, and site.

        :param frame_rate: (int, optional) Frame rate for the video (fps).
        :param image_type: (str, optional) Type of images to include (e.g., 'bright', 'snap').
        :param video_name: (str, optional) Output video file name.
        :param videoDir: (str, optional) Output video location.
        :param camera: (str, optional) Camera identifier to filter by.
        :param site: (str, optional) Site identifier to filter by.
        :param start_time (datetime, optional) Start time for videos - will find closest image
        :param end_time (datetime, optional) End time for videos - will find closest image
        """
        for img_path in self.images.keys():
            if isinstance(self.images, list) or not self.images[img_path]:
                self.initialize_image_handlers()
            continue
        images_metadata = self.get_image_metadata_by_type([image_type], site=site, camera=camera)
        print(images_metadata)
        if not images_metadata:
            print(f"No images found for type '{image_type}' with specified filters.")
            return
        
        # Sort images by timestamp
        images_metadata.sort(key=lambda x: x["timestamp"])
        if start_time or end_time:
            images_metadata = [img for img in images_metadata
                                if (not start_time or img["timestamp"] >= start_time) and
                                (not end_time or img["timestamp"] <= end_time)
                            ]

        if not images_metadata:
            print(f"No images found in the specified time range.")
            return
        # Read the first image to get dimensions
        first_image = cv2.imread(images_metadata[0]["path"])
        height, width, layers = first_image.shape

        # Initialize video writer
        if not video_name:
            video_name = f"{images_metadata[0]['site']}_c{images_metadata[0]['camera']}_{image_type}.mp4"
            
        videoDir = self.config.get('videoDir', os.path.join(os.getcwd(), 'video'))
        os.makedirs(videoDir, exist_ok=True)    
        video_path = os.path.join(videoDir, video_name)
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Existing video found, overwriting: {video_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        # Process each image
        for img_metadata in images_metadata:
            print(img_metadata['path'])
            image = cv2.imread(img_metadata["path"])
            try:
                timestamp = img_metadata["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                text = f"Site: {img_metadata['site']} | Camera: {img_metadata['camera']} | {timestamp}"
            except:
                text = f"Site: {img_metadata['site']} | Camera: {img_metadata['camera']} | {img_metadata['name'].split('.')[0]}"

            # Overlay text on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (20, height - 20), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

            # Write frame to video
            out.write(image)

        # Release video writer
        out.release()
        print(f"Video created successfully: {video_name}")

    def process_image_pairs(self, make_plots=False, save_flag = True):
        """
        Identifies and processes pairs of 'bright' and 'timex' images.
        If both exist, it processes the 'bright' image first and passes the results to 'timex'.
        """
        for img_path in self.images.keys():
            if any(img_type in Path(img_path).stem for img_type in ['bright', 'timex']) and not self.images[img_path]:
                self.images[img_path] = ImageHandler(imagePath = img_path, configPath = self.config)

        image_groups = {}
        
        # Group images by their base name (excluding image type)
        for image_path, handler in self.images.items():
            if handler and handler.image_type in ['bright', 'timex']:
                base_name = handler.image_name.rsplit('.', 1)[0]  # Remove the image type
                if base_name not in image_groups:
                    image_groups[base_name] = {}
                image_groups[base_name][handler.image_type] = handler
        
        # Process pairs of images
        for base_name, images in image_groups.items():
            bright_handler = images.get('bright')
            timex_handler = images.get('timex')
            
            bright_results = None
            
            if bright_handler:
                bright_handler.process_bright(make_plots=make_plots)
                bright_results = bright_handler.processing_results.get('shoreline', {})
                bright_handler.rectify_shoreline()
                if save_flag:
                    bright_handler.save_to_netcdf()
            
            if timex_handler:
                timex_handler.process_timex(bright_coords=bright_results, make_plots=make_plots)
                timex_handler.rectify_shoreline()
                if save_flag:
                    timex_handler.save_to_netcdf()

# ------------- timestack stuff
    def get_runup_from_timestacks(self, segformer_flag = True, save_flag = True):

        for img_path in self.images.keys():
            if 'transect' in Path(img_path).stem and not self.images[img_path]:
                self.images[img_path] = ImageHandler(imagePath = img_path, configPath = self.config)
                
        transect_images = {img_path: handler for img_path, handler in self.images.items() if 'transect' in handler.image_name}

        self.runup = {}

        for img_path, img_handler in transect_images.items():
            print(f"Processing: {img_path}.")
            
            # Run the segformer on the timestack and get the runup from segformer
            if segformer_flag is True:
                print('Running Segformer')
                img_handler.run_segformer_on_timestack()
            print('Extracting runup')
            img_handler.get_runup_from_segformer()
            #img_handler.compute_runup()
            if save_flag is True:
                img_handler.save_to_netcdf()


