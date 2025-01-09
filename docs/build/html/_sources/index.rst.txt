.. ShoreScan documentation master file, created by
   sphinx-quickstart on Thu Jan  9 09:03:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ShoreScan documentation
=======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   methodology

Main Function
=============

The `main` script orchestrates the overall process by integrating the three classes. It initializes an `ImageDatastore` instance, loads images, and applies filters to exclude poor-quality data. Users must specify whether to process all images or only new ones by responding to a prompt (all/new). If they choose new, the script removes already processed images by checking for existing output files (requires both the image and txt file - this can be changed in the future).

After filtering, the script prompts the user to decide whether to enable intermediate plots (yes/no). This option affects how the `ShorelineWorkflow` visualizes intermediate steps. The script then processes images of type bright and timex, passing the filtered datasets and user preferences to the process_images function. Users do not need to specify thresholds directly, as these are hardcoded in filtering and workflow methods.

In summary, the main script provides a user-friendly interface for controlling the pipeline. It automates the flow from image loading to data storage, with minimal input required beyond initial choices. Hardcoded thresholds are spread across the filters and workflows, ensuring consistent processing while allowing flexibility for future refinements.

.. autofunction:: main.main

ImageDatastore
=============

The `ImageDatastore` class is responsible for managing coastal image data, structured hierarchically by site, camera, year, month, day, time, and image type. It provides functionality to load image metadata, filter low-quality images, and generate summary statistics. The images are identified using metadata extracted from their filenames, such as timestamps, site identifiers, and camera labels. This metadata is organized in a nested dictionary, allowing efficient storage and retrieval of specific subsets of images.

A key feature of the class is its filtering functionality. Methods like `filter_black_images`, `filter_white_images`, and `filter_blurry_images` help remove unsuitable images. For example, `filter_black_images` excludes images that are too dark based on a brightness threshold (default: 50), while `filter_white_images` excludes overly bright images (threshold: 200). The method `filter_blurry_images` uses a Laplacian variance threshold (default values depend on image type, e.g., 20 for snap) to detect blurriness. These thresholds are hardcoded but customizable by the user through arguments.

The `ImageDatastore` class also supports copying images into a specified folder, either hierarchically or in a flat structure. Additionally, the `image_stats` method generates useful summaries, such as the total number of images per site or camera, and the distribution of image types. Users can customize some behaviors, such as choosing a root folder for images, but most functionality is predefined for handling typical coastal image datasets. This class ensures that only relevant, high-quality images are passed to subsequent processing stages, enhancing computational efficiency and output reliability.

ShorelineDatastore
=============
The `ShorelineDatastore` class acts as a centralized storage for processed shoreline data. It stores results in a nested dictionary keyed by site, camera, date, and image type. Each entry contains computed shoreline coordinates, bottom boundaries from the segmented SAM mask, watershed segmentation outputs, y-distances, and RMSE values. These metrics are essential for analyzing shoreline dynamics, validating the extraction process, and identifying potential issues.

Users can retrieve stored results for specific images using methods like `get_shoreline_results`. Additionally, the class allows saving shoreline coordinates to text files via `save_shoreline_coords_to_file`. The output filenames are automatically generated based on the metadata of the corresponding image, ensuring easy traceability. This feature enables further analysis or integration with external tools.

Although the `ShorelineDatastore` class does not directly interact with raw image data, it plays a crucial role in organizing and preserving the results of shoreline extraction workflows. Its design ensures consistency and reproducibility, making it a robust component for managing processed data.

ShorelineWorkflow
=============
The `ShorelineWorkflow` class encapsulates the logic for processing individual images and extracting shoreline features. It uses the SAM (Segment Anything Model) for image segmentation, focusing on surf zones. The workflow is tailored to specific image types, such as bright and timex, with separate methods implementing customized processing steps. For example, bright images undergo boundary extraction, watershed segmentation, and RMSE computation.

Several intermediate plots can be generated during processing, such as overlays of detected boundaries and segmentation masks. This is controlled by the `make_intermediate_plots` argument, which users can toggle when initializing the workflow. Hardcoded thresholds exist for filtering irrelevant regions and computing metrics like y-distance and RMSE. For instance, points where the distance between the SAM shoreline and the determined watershed shoreline exceeds 30 pixels are excluded from the final shoreline.

The workflow outputs its results directly to the `ShorelineDatastore`. Each processing step is designed to handle edge cases, such as missing or invalid data, ensuring robustness. Users primarily interact with the workflow through the main script, where they can specify the image type and enable plotting.

