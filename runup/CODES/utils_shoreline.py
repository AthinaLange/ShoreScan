"""
utils_shoreline.py

This module provides functions to extract shorelines from ARGUS-style image products.

"""
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from scipy import interpolate

from segment_anything import SamPredictor, sam_model_registry

# Function to extract random coords from the largest connected component (surfzone point)
def find_surfzone_coords(image_path, num_points = 5, step = 200, max_attempts = 100, make_plot = False):
    """
    Extract up to `num_points` random points from the largest connected component in the image
    representing the surfzone region (typically the shoreline).

    :param image_path: (str) Path to the input image.
    :param num_points: (int, optional) Number of points to find (default is 5).
    :param step: (int, optional) Horizontal step between searches for valid points (default is 200).
    :param max_attempts: (int, optional) Maximum number of attempts to find valid points (default is 100).
    :param make_plot: (bool, optional) Whether to plot intermediate results (default is False).

    :return: marker_coords (list of tuples): List of (x, y) coordinates of the marker points.
    :return: main_region_mask (np.ndarray): Binary mask of the cleaned region.
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
def load_and_predict_sam_model(image_path, checkpoint_path = "segment-anything-main/sam_vit_h_4b8939.pth", model_type = "vit_h", shoreline_coords = [(1000, 300)], beach_coords = None):
    """
    Loads the SAM model and makes a prediction for a given image.

    :param image_path: (str) Path to the image to segment.
    :param checkpoint_path: (str, optional) Path to the SAM model checkpoint (default is "segment-anything-main/sam_vit_h_4b8939.pth").
    :param model_type: (str, optional) Type of SAM model to use ("vit_b", "vit_l", "vit_h") (default is "vit_h").
    :param shoreline_coords: (list of tuples, optional) List of shoreline coordinates (x, y), used as input points (default is [(1000, 300)]).
    :param beach_coords: (list of tuples, optional) Optional list of additional coordinates to help with segmentation (default is None).

    :return: (np.ndarray) The best mask prediction from the SAM model.
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
def extract_bottom_boundary_from_mask(mask, make_plot = False, image_path = ''):
    """
    Extracts the bottom boundary (maximum y-value for each x) from a binary mask.

    :param mask: (np.ndarray) The binary mask where 255 represents the object of interest.
    :param make_plot: (bool, optional) Whether to plot intermediate results (default is False).
    :param image_path: (str, optional) Path to the image to segment (default is '').

    :return: (np.ndarray) The bottom boundary as (x, y) coordinates.
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
def apply_watershed(image_path, bottom_boundary, kernel_size = 200, window_size = 25, make_plot = False):
    """
    Apply watershed segmentation to clean up the shoreline using the bottom boundary as a marker.

    :param image_path: (str) Path to the input image.
    :param bottom_boundary: (np.ndarray) Extracted bottom boundary coordinates used as markers for the watershed algorithm.
    :param kernel_size: (int, optional) Size of the kernel for Gaussian blur (default is 200).
    :param window_size: (int, optional) Size of the sliding window to calculate noise (standard deviation) (default is 25).
    :param make_plot: (bool, optional) Whether to plot intermediate results (default is False).
    
    :return: markers (np.ndarray): The segmented image after watershed with marked boundary.
    :return: watershed_coords (np.ndarray): Extracted coordinates of the watershed boundary.
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

def resample_to_boundary(coords_1, coords_2):
    """
    Resample coordinates from one set to match the length and shape of another using linear interpolation.
    
    :param coords_1: (np.ndarray) First set of coordinates (e.g., watershed boundary).
    :param coords_2: (np.ndarray) Second set of coordinates (e.g., bottom boundary).
        
    :return: (np.ndarray) Resampled coordinates with the same length as the second set.
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

def compute_rmse(coords_1, coords_2):
    """
    Compute the RMSE (Root Mean Squared Error) between two sets of coordinates.
    
    :param coords_1: (np.ndarray) First set of coordinates (e.g., watershed coordinates).
    :param coords_2: (np.ndarray) Second set of coordinates (e.g., bottom boundary).
        
    :return: (float) The RMSE value between the two sets of coordinates, or None if there are issues with the input.
    """
    # Check if both coordinates are 2D and have shape (n, 2)
    if len(coords_1) == 0 or len(coords_2) == 0 or coords_1.ndim != 2 or coords_2.ndim != 2 or coords_1.shape[1] != 2 or coords_2.shape[1] != 2:
        print("Both coords_1 and coords_2 should be 2D arrays with shape (n, 2).")
        return None

    # Resample coords_1 to match coords_2's shape/length
    resampled_coords_1 = resample_to_boundary(coords_1, coords_2)
    
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

def compute_y_distance(coords_1, coords_2):
    """
    Compute the distance between the y-values of two sets of coordinates for each x-coordinate.
    
    :param coords_1: (np.ndarray) First set of coordinates (e.g., watershed boundary).
    :param coords_2: (np.ndarray) Second set of coordinates (e.g., bottom boundary).
        
    :return: (np.ndarray) An array of distances between the y-values of the two coordinate sets.
    """
    # Resample coords_1 to match coords_2
    resampled_coords_1 = resample_to_boundary(coords_1, coords_2)
    
    # Calculate the absolute differences in y-values
    y_distance = np.abs(resampled_coords_1[:, 1] - coords_2[:, 1])
    
    return y_distance

def generate_random_coords_above_line(coords, max_range = 200, min_points = 5, min_y_offset = 100, max_y_offset = 200):
    """
    Generate random coords above a line, with random x-values within dynamically determined intervals.

    :param coords: (np.ndarray or list) (x, y) coords defining the line.
    :param max_range: (int, optional) Maximum range between x-values for each interval. Default is 200.
    :param min_points: (int, optional) Minimum number of intervals (or points) to generate. Default is 5.
    :param min_y_offset: (int, optional) Minimum y-offset to sample above the line. Default is 100.
    :param max_y_offset: (int, optional) Maximum y-offset to sample above the line. Default is 200.
        
    :return: (list) List of (x, y) tuples representing the random coords above the line.
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

def plot_image_and_shoreline(image_path, shoreline_coords = None, watershed_coords = None, other_coords = None, y_distance = None, save_dir = None):
    """
    Plot the image with the bottom boundary and the watershed fitted shoreline.
    
    :param image_path: (str) The path to the image.
    :param shoreline_coords: (np.ndarray, optional) The extracted bottom boundary coords to plot (default is None).
    :param watershed_coords: (np.ndarray, optional) The extracted watershed boundary coords to plot (default is None).
    :param other_coords: (np.ndarray, optional) Any other set of coordinates to plot (default is None).
    :param y_distance: (np.ndarray, optional) The y-distance to plot on a secondary y-axis (default is None).
    :param save_dir: (str, optional) Directory to save the plot (default is None, meaning the plot will be displayed but not saved).
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
    rmse_value = compute_rmse(watershed_coords, shoreline_coords)
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

    title = f"Shoreline and Watershed - {Path(image_path).stem}"
    plt.title(title)

    # Optionally save the plot to the specified directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(save_dir, f"{Path(image_path).stem}.shoreline_plot.png")
        plt.savefig(output_file)
        print(f"Figure saved to {output_file}")
        plt.close(fig)
    else:
        # Show the plot
        plt.show()

