import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
import os
from scipy import interpolate
import matplotlib.pyplot as plt
import random

class ShorelineWorkflow:
    def __init__(self, image_path, image_type, shoreline_datastore, make_intermediate_plots = False):
        """
        Initialize the workflow for processing a single image.

        Parameters:
        - image_path: The path to the image to process.
        - image_type: The type of the image (e.g., 'bright', 'timex', etc.).
        - shoreline_datastore: The datastore to save results.
        - make_intermediate_plots: Boolean to toggle intermediate plots.
        """
        self.image_path = image_path  # Path to the image to process
        self.image_type = image_type  # Type of the image ('bright', 'timex', etc.)
        self.shoreline_datastore = shoreline_datastore  # Datastore to store the results
        self.make_intermediate_plots = make_intermediate_plots #Boolean to toggle intermediate plots.
    
    def _show_plot(self, plt_func, *args, **kwargs):
        """
        Helper function to handle intermediate plot display.
        Only shows the plot if `make_intermediate_plots` is True.
        """
        if self.make_intermediate_plots:
            plt_func(*args, **kwargs)
        else:
            plt.close()

    def process(self):
        """
        Main function to process a single image based on the selected `image_type`.
        Calls the appropriate workflow method based on the image type.
        """
        if self.image_type == 'bright':
            return self._process_bright()
        elif self.image_type == 'timex':
            return self._process_timex()
        elif self.image_type == 'dark':
            return self._process_dark()
        elif self.image_type == 'snap':
            return self._process_snap()
        elif self.image_type == 'var':
            return self._process_var()
        else:
            return 'No workflow currently available for this image type.'

    def _process_bright(self):
        """
        Process a single 'bright' image.
        
        This method follows the workflow specific to 'bright' image types. It performs the following steps:
        1. Extract bottom boundaries from the image in three attempts (using SAM).
        2. Apply a watershed segmentation to the image based on SAM shoreline.
        3. Compute RMSE between watershed coordinates and SAM shoreline.
        4. Plot and store the results.

        Returns:
        - A string indicating that the 'bright' image processing is complete.
        """
        print(f"Processing bright image: {self.image_path}")

        # Extract metadata from the image path
        parts = self.image_path.split(os.sep)[-1].split('.')  # Extract filename and split by '.'

        # Ensure that the file has enough parts for extraction
        if len(parts) < 9:
            raise ValueError(f"File does not match the expected format: {self.image_path}")

        # Extract relevant metadata from the filename
        month = parts[2]      # Month
        day = parts[3][:2]    # Day of the month
        time = parts[3][3:]   # Time (HH_MM)
        year = parts[5]       # Year
        site = parts[6]       # Site CACO#
        camera = parts[7]     # Camera (e.g., "c1")
                
        # Initialize arrays to store bottom boundary estimates from three attempts
        bottom_boundaries = []
        for _ in range(3):
            # Attempt to find surfzone points and predict using SAM model
            coords, _ = self.find_surfzone_coords(self.image_path, num_points = 5, make_plot = self.make_intermediate_plots)
            best_mask = self.load_and_predict_sam_model(self.image_path, shoreline_coords = coords)
            # Extract the bottom boundary from the mask
            bottom_boundary = self.extract_bottom_boundary_from_mask(best_mask, make_plot = self.make_intermediate_plots, image_path = self.image_path)
            bottom_boundaries.append(bottom_boundary)

        # Convert bottom boundary attempts into a NumPy array and compute the median bottom boundary
        bottom_boundaries = np.array(bottom_boundaries)  # Shape: (3, n, 2)
        bottom_boundary_median = np.median(bottom_boundaries, axis = 0)  # Shape: (n, 2)

        # Apply watershed segmentation to the image using the median bottom boundary
        _, watershed_coords = self.apply_watershed(self.image_path, bottom_boundary_median, make_plot = self.make_intermediate_plots)

        # Clean the boundary coords by removing out-of-bounds values based on mask coordinates
        # mask_x = np.any(best_mask, axis=0)  # Check columns (x-values) for non-zero values
        # x_min = np.argmax(mask_x)  # First non-zero index
        # x_max = len(mask_x) - 1 - np.argmax(mask_x[::-1])  # Last non-zero index

        # Remove coords outside of the x-min to x-max range
        # bottom_boundary_median[(bottom_boundary_median[:, 0] < x_min) | (bottom_boundary_median[:, 0] > x_max), 1] = np.nan
        # watershed_coords[(watershed_coords[:, 0] < x_min) | (watershed_coords[:, 0] > x_max), 1] = np.nan

        # Compute the y-distance and RMSE between the watershed coordinates and the bottom boundary
        y_distance = self.compute_y_distance(watershed_coords, bottom_boundary_median)
        rmse_value = self.compute_rmse(watershed_coords, bottom_boundary_median)
        print(f"RMSE: {np.round(rmse_value, 2)} pixels")

        # Clean shoreline coords based on distance threshold
        shoreline_coords = bottom_boundary_median.copy()
        shoreline_coords[y_distance > 30, :] = np.nan  # Set points beyond 30 pixels to NaN

        # Plot the image and overlay the shoreline and watershed coords
        self.plot_image_and_shoreline(self.image_path, shoreline_coords = shoreline_coords, watershed_coords = watershed_coords, y_distance = y_distance, save_dir = 'shoreline_plots/')
        plt.close()

        # Store the results in the datastore
        self.shoreline_datastore.store_shoreline_results(
            site = site,
            camera = camera,
            year = year,
            month = month,
            day = day,
            time = time,
            image_type = self.image_type,
            shoreline_coords = shoreline_coords,
            bottom_boundary = np.vstack([bottom_boundaries[0][:, 0], bottom_boundaries[0][:, 1], bottom_boundaries[1][:, 1], bottom_boundaries[2][:, 1], bottom_boundary_median[:, 1]]).T,
            watershed_coords = watershed_coords,
            y_distance = y_distance,
            rmse_value = rmse_value
        )
        print("Bright image processing complete.")
        return self.shoreline_datastore
    
    def _process_timex(self):
        """
        Process a single 'timex' image.
        
        This method follows the workflow specific to 'timex' image types. It uses the results from the 'bright' image 
        processing and follows similar steps to the 'bright' workflow.

        Returns:
        - A string indicating that the 'timex' image processing is complete.
        """
        print(f"Processing timex image: {self.image_path}")

        # Extract metadata from the image path
        parts = self.image_path.split(os.sep)[-1].split('.')  # Extract filename and split by '.'

        # Ensure that the file has enough parts for extraction
        if len(parts) < 9:
            raise ValueError(f"File does not match the expected format: {self.image_path}")

        # Extract relevant metadata from the filename
        month = parts[2]      # Month
        day = parts[3][:2]    # Day of the month
        time = parts[3][3:]   # Time (HH_MM)
        year = parts[5]       # Year
        site = parts[6]       # Site CACO#
        camera = parts[7]     # Camera (e.g., "c1")
        
        # Retrieve corresponding shoreline coords from previously processed 'bright' images
        bright_coords = self.shoreline_datastore.get_shoreline_coords(
            site = site,
            camera = camera,
            year = year,
            month = month,
            day = day,
            time = time,
            image_type = 'bright'  # We assume 'bright' images have been processed previously
        )
        # Initialize an array to store bottom boundaries
        bottom_boundaries = []
        for _ in range(3):
            # Generate random points above the bright boundary
            if bright_coords is not None:
                water_coords = self.generate_random_coords_above_line(bright_coords[0], max_range = 200, min_points = 5, min_y_offset = 100, max_y_offset = 350)
            else:
                water_coords = [(1000, 300)]  # Default coords if no bright data is available

            # Predict the bottom boundary using SAM model
            best_mask = self.load_and_predict_sam_model(self.image_path, shoreline_coords = water_coords)
            bottom_boundary = self.extract_bottom_boundary_from_mask(best_mask, make_plot = self.make_intermediate_plots, image_path = self.image_path)
            bottom_boundaries.append(bottom_boundary)

        # Convert bottom boundaries into a NumPy array and compute the median bottom boundary
        bottom_boundaries = np.array(bottom_boundaries)
        bottom_boundary_median = np.median(bottom_boundaries, axis=0)

        shoreline_coords = bottom_boundary_median.copy()
        # Apply watershed segmentation using the median bottom boundary
        _, watershed_coords = self.apply_watershed(self.image_path, bottom_boundary_median, make_plot = self.make_intermediate_plots)

        # Compute y-distance and RMSE if there are valid watershed coordinates
        if len(watershed_coords) != 0:
            y_distance = self.compute_y_distance(watershed_coords, shoreline_coords)
            rmse_value = self.compute_rmse(watershed_coords, shoreline_coords)
            print(f"RMSE: {np.round(rmse_value, 2)} pixels")

            # Remove coords where y-distance exceeds threshold
            shoreline_coords[y_distance > 10, :] = np.nan
        else:
            rmse_value = np.nan
            y_distance = None

        # Plot and store results
        if bright_coords is not None:
            self.plot_image_and_shoreline(self.image_path, shoreline_coords = shoreline_coords, watershed_coords = watershed_coords, other_coords = bright_coords[0], y_distance = y_distance, save_dir = 'shoreline_plots/')
        else:
            self.plot_image_and_shoreline(self.image_path, shoreline_coords = shoreline_coords, watershed_coords = watershed_coords, y_distance = y_distance, save_dir = 'shoreline_plots/')

        # Store the processed results in the datastore
        self.shoreline_datastore.store_shoreline_results(
            site = site,
            camera = camera,
            year = year,
            month = month,
            day = day,
            time = time,
            image_type = self.image_type,
            shoreline_coords = shoreline_coords,
            bottom_boundary = np.vstack([bottom_boundaries[0][:, 0], bottom_boundaries[0][:, 1], bottom_boundaries[1][:, 1], bottom_boundaries[2][:, 1], bottom_boundary_median[:, 1]]).T,
            watershed_coords = watershed_coords,
            y_distance = y_distance,
            rmse_value = rmse_value
        )
        print("Timex image processing complete.")
        return self.shoreline_datastore

    def _process_dark(self):
        """
        Process a single 'dark' image. Placeholder for future implementation.
        """
        return "No workflow for dark image type yet. Please select bright or timex."
    
    def _process_snap(self):
        """
        Process a single 'snap' image. Placeholder for future implementation.
        """
        return "No workflow for snap image type yet. Please select bright or timex."
    
    def _process_var(self):
        """
        Process a single 'var' image. Placeholder for future implementation.
        """
        return "No workflow for var image type yet. Please select bright or timex."

    # Function to extract random coords from the largest connected component (surfzone point)
    @staticmethod
    def find_surfzone_coords(image_path, num_points = 5, step = 200, max_attempts = 100, make_plot = False):
        """
        Extract up to `num_points` random points from the largest connected component in the image
        representing the surfzone region (typically the shoreline).

        Parameters:
            image_path (str): Path to the input image.
            num_points (int): Number of points to find.
            step (int): Horizontal step between searches for valid points.
            max_attempts (int): Maximum number of attempts to find valid points.
            make_plot (bool): Whether to plot intermediate results.

        Returns:
            marker_coords (list): List of (x, y) coordinates of the marker coords.
            main_region_mask (np.ndarray): Binary mask of the cleaned region.
        """
        # Load and preprocess the image (only the first channel as grayscale)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, 0]

        # Threshold the image to isolate the surfzone region (assumed to be white)
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find connected components and select the largest one
        _, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
        largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Create a mask for the largest component
        largest_component_mask = (labels == largest_component_label).astype(np.uint8)

        # Clean the mask with morphological operations (open and erode)
        kernel = np.ones((25, 100), np.uint8)
        cleaned_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_OPEN, kernel)
        erosion_mask = cv2.erode(cleaned_mask, kernel)

        # Re-run connected components on the cleaned mask to isolate the largest region
        _, labels, stats, _ = cv2.connectedComponentsWithStats(erosion_mask, connectivity=8)
        largest_cleaned_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_region_mask = (labels == largest_cleaned_label).astype(np.uint8)

        # Initialize the point search process
        h, w = main_region_mask.shape
        attempts = 0
        marker_coords = []

        # Retry until enough points are found or max_attempts is reached
        while len(marker_coords) < num_points and attempts < max_attempts:
            marker_coords = []
            attempts += 1
            for x in range(0, w, step):
                # Find valid y-coordinates within the horizontal range
                x_range = main_region_mask[:, x:x+step]
                valid_y, valid_x = np.where(x_range == 1)
                if len(valid_y) > 0:
                    # Randomly select a valid point from this range
                    idx = random.randint(0, len(valid_y) - 1)
                    y = valid_y[idx]
                    x_center = x + valid_x[idx]  # Account for offset in x-range
                    marker_coords.append((x_center, y))

        if len(marker_coords) < num_points:
            print(f"Failed to find enough points after {max_attempts} attempts.")

        # Display the results for visual inspection
        if make_plot:
            plt.figure(figsize = (10, 10))
            plt.imshow(image, cmap = 'gray')
            plt.imshow(main_region_mask, alpha = 0.3)  # Overlay the mask
            for (x, y) in marker_coords:
                plt.plot(x, y, 'ro')  # Plot marker coords as red circles
            plt.title('Marker Points Every 200 Pixels with Random Y')
            plt.show()

        return marker_coords, main_region_mask

    # Function to load and predict using SAM model
    @staticmethod
    def load_and_predict_sam_model(image_path, checkpoint_path = "segment-anything-main/sam_vit_h_4b8939.pth", model_type = "vit_h", shoreline_coords = [(1000, 300)], beach_coords = None):
        """
        Loads the SAM model and makes a prediction for a given image.

        Parameters:
            image_path (str): Path to the image to segment.
            checkpoint_path (str): Path to the SAM model checkpoint.
            model_type (str): Type of SAM model to use ("vit_b", "vit_l", "vit_h").
            shoreline_coords (list): List of shoreline coordinates (x, y).
            beach_coords (list): Optional list of additional coords to help with segmentation.
         
        Returns:
            np.ndarray: The best mask prediction from the SAM model.
        """
        # Load the SAM model and set the device (cuda if available)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)

        # Load the image
        image = np.array(Image.open(image_path))

        # Prepare the SAM predictor
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        # Prepare the coords and labels for SAM prediction
        shoreline_point_coords = np.array(shoreline_coords)
        shoreline_labels = np.ones(len(shoreline_coords), dtype = np.int32)  # All coords are labeled as 1 (foreground)

        if beach_coords is not None:
            beach_point_coords = np.array(beach_coords)
            beach_labels = np.full(len(beach_coords), 2, dtype = np.int32)  # Label beach coords as 2

            # Concatenate the coords and labels
            all_point_coords = np.concatenate((shoreline_point_coords, beach_point_coords), axis = 0)
            all_point_labels = np.concatenate((shoreline_labels, beach_labels), axis = 0)
        else:
            all_point_coords = shoreline_point_coords
            all_point_labels = shoreline_labels

        # Predict the masks from SAM
        masks, scores, _ = predictor.predict(
            point_coords = all_point_coords,
            point_labels = all_point_labels,
            multimask_output = True,
        )

        # Select the best mask based on the highest score
        best_mask = masks[np.argmax(scores)]

        return best_mask

    # Function to extract bottom boundary (maximum y-value for each x) from the mask
    @staticmethod
    def extract_bottom_boundary_from_mask(mask, make_plot = False, image_path = ''):
        """
        Extracts the bottom boundary (maximum y-value for each x) from a binary mask.

        Parameters:
            mask (np.ndarray): The binary mask where 255 represents the object of interest.
            make_plot (bool): Whether to plot intermediate results.
            image_path (str): Path to the image to segment.
        
        Returns:
            np.ndarray: The bottom boundary as (x, y) coordinates.
        """
        # Ensure the mask is binary (0 or 255)
        binary_mask = (mask > 0).astype(np.uint8)
        mask_width = mask.shape[1]
        
        # Dictionary to store the bottom-most y-coordinate for each x-coordinate
        bottom_boundary = {}

        # Loop over each column (x coordinate)
        for x in range(binary_mask.shape[1]):
            y_points = np.where(binary_mask[:, x] > 0)[0]  # y coordinates where the mask is non-zero
            
            if len(y_points) > 0:
                max_y = np.max(y_points)
                bottom_boundary[x] = max_y

        # Convert the dictionary to a sorted array of (x, y) tuples
        sorted_bottom_boundary = sorted(bottom_boundary.items())
        bottom_boundary_coords = np.array(sorted_bottom_boundary)

        # Interpolate y-coordinates to smooth the bottom boundary
        new_x_points = np.linspace(0, mask_width-1, mask_width)
        interpolated_y_points = np.interp(new_x_points, bottom_boundary_coords[:, 0], bottom_boundary_coords[:, 1])

        # Combine the new x and y coordinates to form the interpolated bottom boundary
        bottom_boundary_coords = np.column_stack((new_x_points, interpolated_y_points))

        if make_plot:
            # Load the image
            image = np.array(Image.open(image_path))

            plt.figure(figsize = (10, 10))
            plt.imshow(image, alpha = 1.0)  # Original image
            plt.imshow(mask, cmap = 'jet', alpha = 0.5)  # Overlay mask with transparency
            plt.scatter(bottom_boundary_coords[:,0], bottom_boundary_coords[:,1], c = 'black')
            plt.title("Original Image with Predicted Mask Overlay")
            plt.axis("off")
            plt.show()

        return bottom_boundary_coords

    # Function to apply watershed algorithm for shoreline segmentation
    @staticmethod
    def apply_watershed(image_path, bottom_boundary, kernel_size = 200, window_size = 25, make_plot = False):
        """
        Apply watershed segmentation to clean up the shoreline using the bottom boundary as a marker.

        Parameters:
            image_path (str): Path to the input image.
            bottom_boundary (np.ndarray): Extracted bottom boundary coords.
            kernel_size (int): Size of the kernel for Gaussian blur.
            window_size (int): Size of the sliding window to calculate noise (standard deviation).
            make_plot (bool): Whether to plot intermediate results.
        
        Returns:
            np.ndarray: The segmented image after watershed with marked boundary.
            np.ndarray: Watershed markers used for segmentation.
            np.ndarray: Extracted coordinates of the watershed boundary.
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Initialize markers for watershed segmentation (background is 0)
        markers = np.zeros_like(image[:, :, 0], dtype=np.int32)

        # Extract x and y coordinates from the bottom boundary
        bottom_boundary = bottom_boundary.astype(np.int32)
        x_points, y_points = bottom_boundary[:, 0], bottom_boundary[:, 1]

        # Initialize array for dynamic offsets based on local standard deviation
        dynamic_offsets = np.zeros(len(y_points))
        for i in range(len(y_points)):
            start = max(0, i - window_size // 2)
            end = min(len(y_points), i + window_size // 2)
            window = y_points[start:end]

            # Compute local standard deviation (scaled for offset adjustment)
            local_std = np.exp(np.mean(np.abs(np.gradient(window))))
            dynamic_offsets[i] = max(5, int(5 + local_std))  # Minimum offset of 5

        # Smooth offsets with a moving average kernel
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_offsets = np.convolve(np.pad(dynamic_offsets, kernel_size//2, mode = 'reflect'), kernel, mode = 'valid')
        smoothed_y = np.convolve(np.pad(y_points, kernel_size//2, mode = 'reflect'), kernel, mode = 'valid')
        #!!! Should be looked at later
        smoothed_y = smoothed_y[:len(x_points)]
        smoothed_offsets = smoothed_offsets[:len(x_points)]
        # Generate new smoothed boundary coordinates using dynamic offsets
        smoothed_bottom_boundary_above = np.column_stack((x_points, smoothed_y - smoothed_offsets))
        smoothed_bottom_boundary_below = np.column_stack((x_points, smoothed_y + smoothed_offsets))

        # Place markers based on the smoothed boundaries
        for (x_above, y_above), (x_below, y_below) in zip(smoothed_bottom_boundary_above, smoothed_bottom_boundary_below):
            if y_above - 10 > 0:
                markers[:max(0, int(y_above)), int(x_above)] = 1  # Water marker (above boundary)
            if y_below + 10 < image.shape[0]:
                markers[min(image.shape[0] - 1, int(y_below)):, int(x_below)] = 2  # Sand marker (below boundary)

        # Apply distance transform to the markers
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(markers.astype(np.uint8)), cv2.DIST_L2, 5)
        _, thresh = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Apply watershed algorithm
        markers = np.int32(markers)
        markers[thresh == 0] = -1  # Mark unknown areas as -1
        markers = cv2.watershed(image, markers)

        # Remove boundaries from the borders
        markers[0, :], markers[-1, :], markers[:, 0], markers[:, -1] = 0, 0, 0, 0
        
        # Extract watershed boundary coordinates
        y_points, x_points = np.where(markers == -1)

        # Extract and sort boundary coordinates
        boundary_coords = np.column_stack((x_points, y_points))
        sorted_boundary_coords = boundary_coords[np.argsort(boundary_coords[:, 0])]
        _, unique_indices = np.unique(sorted_boundary_coords[:, 0], return_index = True)
        unique_boundary_coords = sorted_boundary_coords[unique_indices]

        # Ensure a consistent format for the output coordinates
        unique_x_points, unique_y_points = unique_boundary_coords[:, 0], unique_boundary_coords[:, 1]
        if len(unique_x_points) != 0:
            unique_x_points = np.concatenate(([0], unique_x_points, [unique_x_points[-1] + 1]))
            unique_y_points = np.concatenate(([np.nan], unique_y_points, [np.nan]))

        watershed_coords = np.column_stack((unique_x_points, unique_y_points))

        if make_plot:
            plt.figure(figsize = (10, 10))
            plt.imshow(image, cmap = 'gray')
            # Visualize watershed boundaries in red
            image[markers == -1] = [0, 0, 255]
            plt.imshow(image, alpha = 0.3)  # Overlay the mask
            for (x, y) in watershed_coords:
                plt.plot(x, y, 'ro')  # Plot marker coords as red circles
            plt.title('Watershed Algorithm')
            plt.show()

        return markers, watershed_coords

    @staticmethod
    def resample_to_boundary(coords_1, coords_2):
        """
        Resample coordinates from one set to match the length and shape of another using linear interpolation.
        
        Parameters:
            coords_1 (np.ndarray): First set of coordinates (e.g., watershed boundary).
            coords_2 (np.ndarray): Second set of coordinates (e.g., bottom boundary).
            
        Returns:
            np.ndarray: Resampled coordinates with the same length as the second set.
        """
        # Convert to numpy arrays if necessary
        coords_1 = np.array(coords_1)
        coords_2 = np.array(coords_2)
        
        # Linear interpolation for x and y coordinates
        x_interp = interpolate.interp1d(np.linspace(0, 1, len(coords_1)), coords_1[:, 0], kind = 'linear', fill_value = "extrapolate")
        y_interp = interpolate.interp1d(np.linspace(0, 1, len(coords_1)), coords_1[:, 1], kind = 'linear', fill_value = "extrapolate")
        
        # Resample the coordinates to match the second set's length
        resampled_x = x_interp(np.linspace(0, 1, len(coords_2)))
        resampled_y = y_interp(np.linspace(0, 1, len(coords_2)))
        
        return np.column_stack((resampled_x, resampled_y))

    @staticmethod
    def compute_rmse(coords_1, coords_2):
        """
        Compute the RMSE (Root Mean Squared Error) between two sets of coordinates.
        
        Parameters:
            coords_1 (np.ndarray): First set of coordinates (e.g., watershed coordinates).
            coords_2 (np.ndarray): Second set of coordinates (e.g., bottom boundary).
            
        Returns:
            float: The RMSE value between the two sets of coordinates, or None if there are issues with the input.
        """
        # Check if both coordinates are 2D and have shape (n, 2)
        if len(coords_1) == 0 or len(coords_2) == 0 or coords_1.ndim != 2 or coords_2.ndim != 2 or coords_1.shape[1] != 2 or coords_2.shape[1] != 2:
            print("Both coords_1 and coords_2 should be 2D arrays with shape (n, 2).")
            return None

        # Resample coords_1 to match coords_2's shape/length
        resampled_coords_1 = ShorelineWorkflow.resample_to_boundary(coords_1, coords_2)
        
        # Ensure both coordinates are numpy arrays (2D with x, y coordinates)
        coords_1 = np.array(resampled_coords_1)
        coords_2 = np.array(coords_2)
        
        # Ensure both have the same length after resampling
        if len(coords_1) != len(coords_2):
            print("The number of points in coords_1 and coords_2 must be the same after resampling.")
            return None
        
        # Identify the valid (non-NaN) entries
        valid_mask = ~np.isnan(coords_1[:, 0]) & ~np.isnan(coords_1[:, 1]) & ~np.isnan(coords_2[:, 0]) & ~np.isnan(coords_2[:, 1])

        # Filter out NaN values from both arrays using the valid mask
        coords_1 = coords_1[valid_mask]
        coords_2 = coords_2[valid_mask]

        # Ensure there are valid data points left
        if len(coords_1) == 0:
            print("No valid data points left after filtering NaNs.")
            return None

        # Calculate the squared differences between corresponding points
        diff = coords_1[:, 1] - coords_2[:, 1]
        squared_diff = np.square(diff)
        
        # Calculate the mean squared error (MSE) for both x and y coordinates
        mse = np.mean(squared_diff)
        
        # Calculate RMSE (root of MSE)
        rmse = np.sqrt(mse)
        
        return rmse

    @staticmethod
    def compute_y_distance(coords_1, coords_2):
        """
        Compute the distance between the y-values of two sets of coordinates for each x-coordinate.
        
        Parameters:
            coords_1 (np.ndarray): First set of coordinates (e.g., watershed boundary).
            coords_2 (np.ndarray): Second set of coordinates (e.g., bottom boundary).
            
        Returns:
            np.ndarray: An array of distances between the y-values of the two coordinate sets.
        """
        # Resample coords_1 to match coords_2
        resampled_coords_1 = ShorelineWorkflow.resample_to_boundary(coords_1, coords_2)
        
        # Calculate the absolute differences in y-values
        y_distance = np.abs(resampled_coords_1[:, 1] - coords_2[:, 1])
        
        return y_distance

    @staticmethod
    def generate_random_coords_above_line(coords, max_range = 200, min_points = 5, min_y_offset = 100, max_y_offset = 200):
        """
        Generate random coords above a line, with random x-values within dynamically determined intervals.

        Parameters:
        - coords (np.ndarray or list): (x, y) coords defining the line.
        - max_range (int, optional): Maximum range between x-values for each interval. Default is 200.
        - min_points (int, optional): Minimum number of intervals (or points) to generate. Default is 5.
        - min_y_offset (int, optional): Minimum y-offset to sample above the line. Default is 100.
        - max_y_offset (int, optional): Maximum y-offset to sample above the line. Default is 200.

        Returns:
        - random_coords (list): List of (x, y) tuples representing the random coords above the line.
        """
        
        # Extract x and y values from the line coords
        x_values = coords[:, 0]  # x-coordinates of the line
        y_values = coords[:, 1]  # y-coordinates of the line

        # Create a dictionary for fast lookup of y-values for given x-values
        line_dict = dict(zip(x_values, y_values))

        random_coords = []

        # Determine the total x-range from the line's x-values
        x_min, x_max = int(min(x_values)), int(max(x_values))

        # Calculate the number of intervals based on max_range (with a minimum of one interval)
        num_intervals = (x_max - x_min) // max_range
        num_intervals = max(1, num_intervals)  # Ensure at least one interval
        
        # Ensure there are at least 'min_points' intervals
        if num_intervals < min_points:
            num_intervals = min_points

        # Calculate the width of each interval
        interval_width = (x_max - x_min) / num_intervals

        for i in range(num_intervals):
            # Determine the x-range for the current interval
            interval_start = x_min + i * interval_width
            interval_end = x_min + (i + 1) * interval_width
            
            # Ensure we don't exceed the overall range
            if interval_end > x_max:
                interval_end = x_max
            
            # Randomly pick an x-value within the current interval
            x_random = random.randint(int(interval_start), int(interval_end))

            # Find the closest x-value from the line to use for y-value
            closest_x = min(x_values, key=lambda x: abs(x - x_random))
            y_on_line = line_dict[closest_x]
            
            # Add a random offset to the y-value to place the point above the line
            y_offset = random.uniform(min_y_offset, min(max_y_offset, y_on_line))  # Offset above the line
            y_random = y_on_line - y_offset
            
            # Store the randomly generated point (x_random, y_random)
            random_coords.append((x_random, y_random))

        return random_coords

    @staticmethod
    def plot_image_and_shoreline(image_path, shoreline_coords = None, watershed_coords = None, other_coords = None, y_distance = None, save_dir = None):
        """
        Plot the image with the bottom boundary and the watershed fitted shoreline.
        
        Parameters:
            image_path (str): The path to the image.
            bottom_boundary (np.ndarray): The extracted bottom boundary coords.
            watershed_coords (np.ndarray): The extracted watershed boundary coords.
            y_distance (np.ndarray): The y-distance to plot on a secondary y-axis.
            save_dir (str): Directory to save the plot (optional).
        """
        # Read the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create the figure
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the image on the first axis
        ax1.imshow(image_rgb)
        
        # Plot the bottom boundary and watershed coordinates
        if other_coords is not None:
            ax1.plot(other_coords[:, 0], other_coords[:, 1], c = 'k', label = 'Another Boundary')
        if watershed_coords is not None:
            ax1.plot(watershed_coords[:, 0], watershed_coords[:, 1], c = 'r', label = 'Watershed Boundary')
        if shoreline_coords is not None:
            ax1.plot(shoreline_coords[:, 0], shoreline_coords[:, 1], c = 'g', label = 'Bottom Boundary (SAM)')
        
        # Create the secondary y-axis for y_distance
        ax2 = ax1.twinx()
        # Compute RMSE if both boundaries are provided
        rmse_value = ShorelineWorkflow.compute_rmse(watershed_coords, shoreline_coords)
        if rmse_value is not None:
            lb = f'Y Distance (RMSE = {rmse_value:.2f} pixels)'
        else:
            lb = 'Y Distance'
        
        # Plot y_distance if available
        if y_distance is not None:
            ax2.plot(shoreline_coords[:, 0], y_distance, c = 'b', label = lb, linestyle = '--')
        
        # Set axis labels and title
        ax1.set_xlabel('X Coordinates')
        ax1.set_ylabel('Y Coordinates', color = 'black')
        ax2.set_ylabel('Y Distance', color = 'blue')

        # Set the limits of the y-axes
        ax1.set_ylim(image.shape[0], 0)  # Reverse y-axis to match image orientation
        ax2.set_ylim(0, 50) # Set y-distance limits

        # Set the aspect ratio to auto for better visualization
        ax1.set_aspect('auto')

        # Add legends for both axes
        ax1.legend(loc = 'upper left')
        ax2.legend(loc = 'upper right')

        # Title
        # Extract metadata from the image path
        parts = image_path.split(os.sep)[-1].split('.')  # Extract filename and split by '.'

        # Ensure that the file has enough parts for extraction
        if len(parts) < 9:
            raise ValueError(f"File does not match the expected format: {image_path}")

        # Extract relevant metadata from the filename
        month = parts[2]      # Month
        day = parts[3][:2]    # Day of the month
        time = parts[3][3:]   # Time (HH_MM)
        year = parts[5]       # Year
        site = parts[6]       # Site CACO#
        camera = parts[7]     # Camera (e.g., "c1")
        image_type = parts[8]  # Image type (e.g., "bright", "dark", "timex", "snap")

        file_name = f"{site}_{camera}_{year}_{month}_{day}_{time}_{image_type}"
        title = f"Shoreline and Watershed - {file_name}"
        plt.title(title)
    
        # Optionally save the plot to the specified directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            output_file = os.path.join(save_dir, f"{file_name}_shoreline_plot.png")
            plt.savefig(output_file)
            print(f"Figure saved to {output_file}")
            plt.close(fig)
        else:
            # Show the plot
            plt.show()
