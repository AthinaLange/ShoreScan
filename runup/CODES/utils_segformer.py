# Functions for analyzing Runup SegFormer results
"""
utils_segformer.py

This module provides functions for extracting wave runup from the softmax image given by the SegFormer Runup algorithm.

Dependencies:
    - os
    - re
    - cv2
    - numpy
    - datetime
    - scipy

"""
import os
import re
import cv2
import numpy as np
from datetime import datetime, timezone
import scipy.signal
import scipy.interpolate

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
