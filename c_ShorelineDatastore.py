import numpy as np
from collections import defaultdict
import os

class ShorelineDatastore:
    def __init__(self):
        """
        Initialize the datastore with the root folder containing the images.

        Args:
        - root_folder (str): The path to the root folder containing image files. 
                             If None, a dialog will prompt the user to select a folder.
        """
        # Nested dictionary to store results: site -> camera -> year -> month -> day -> time -> image_type -> data_type -> data
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))))

    def store_shoreline_results(self, site, camera, year, month, day, time, image_type, shoreline_coords, bottom_boundary, watershed_coords, y_distance, rmse_value):
        """
        Store the computed results for a given image.

        Args:
        - site (str): The site identifier.
        - camera (str): The camera identifier.
        - year (str): The year of the image.
        - month (str): The month of the image.
        - day (str): The day of the image.
        - time (str): The time of the image.
        - image_type (str): The type of the image (e.g., 'bright').
        - shoreline_coords (np.ndarray): The final shoreline coords.
        - bottom_boundary (np.ndarray): The computed bottom boundarys. (1,2,3,median)
        - watershed_coords (np.ndarray): The watershed coordinates.
        - y_distance (np.ndarray): The computed y-distance.
        - rmse_value (float): The computed RMSE value.
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

        Args:
        - site (str): The site identifier.
        - camera (str): The camera identifier.
        - year (str): The year of the image.
        - month (str): The month of the image.
        - day (str): The day of the image.
        - time (str): The time of the image.
        - image_type (str): The type of the image (e.g., 'bright').

        Returns:
        - dict: A dictionary containing the shoreline coords, bottom boundary, watershed coords, y-distance, and RMSE values.
        """
        return self.results[site][camera][year][month][day][time][image_type]

    def get_shoreline_coords(self, site, camera, year, month, day, time, image_type):
        """
        Retrieve all filtered bottom boundaries for a given image.

        Args:
        - site (str): The site identifier.
        - camera (str): The camera identifier.
        - year (str): The year of the image.
        - month (str): The month of the image.
        - day (str): The day of the image.
        - time (str): The time of the image.
        - image_type (str): The type of the image (e.g., 'bright').

        Returns:
        - list: A list of all shoreline coords for the given image.
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

        Args:
        - site (str): The site identifier.
        - camera (str): The camera identifier.
        - year (str): The year of the image.
        - month (str): The month of the image.
        - day (str): The day of the image.
        - time (str): The time of the image.
        - image_type (str): The type of the image (e.g., 'bright').
        - output_folder (str): Folder to save the output file. Default is "output".
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