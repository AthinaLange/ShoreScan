Processing Methods
==================

The shoreline extraction process involves several key steps, each contributing to a robust model capable of accurately identifying and processing shoreline points. Here's an overview of the steps involved, including specific values for parameters used in each stage:

**Finding Surfzone Points**: The first step involves extracting up to five random points from the largest connected component within the image, using an Otsu threshold to isolate the surfzone area (the white region). The image is preprocessed by converting it to grayscale and applying a binary threshold. Morphological operations, including opening and erosion with a kernel of size (25, 100), are used to clean the mask. The random points are selected in intervals of 200 pixels along the x-axis. This process is repeated up to 100 times (max attempts) to ensure enough points are found. If successful, the extracted points are used for further segmentation.

**SAM Model Prediction**: After identifying the shoreline points, the next step is to use the Segment Anything Model (SAM) for segmentation. The SAM model is loaded using the specified checkpoint, "segment-anything-main/sam_vit_h_4b8939.pth", and set to use the "vit_h" model type. The model takes the identified surfzone points, labels them as foreground, and outputs a mask. The best mask is selected based on the highest score, which indicates the model's confidence in the prediction. The model is implemented in PyTorch, and it runs on either a CUDA-enabled GPU or CPU depending on the device availability.

**Bottom Boundary Extraction**: Once the mask is generated, the next step is to extract the bottom boundary, which represents the shoreline's lower boundary (maximum y-coordinate for each x-coordinate). This is done by iterating over each x-coordinate and identifying the corresponding y-coordinate where the mask is non-zero. The points are then interpolated to ensure the boundary is continuous and precise.

**Watershed Segmentation**: The final segmentation step involves using the watershed algorithm to refine the boundary. The median bottom boundary points are used as the boundary between sand and water. A dynamic offset from the obtained points based on the exponent of the mean gradient of the window, then smoothed with a Gaussian filter with a kernel size of 200 is used to generate the water markers as above this line and sand markers below. The watershed algorithm is then applied, and the boundary extracted. The result is a set of boundary coordinates that represent an alternative shoreline.

**Evaluation**: To evaluate the accuracy of the extracted shorelines, the root mean square error (RMSE) between the watershed coordinates and the median bottom boundary is computed. Additionally, the y-distance between the two sets of coordinates is calculated to assess how closely the watershed boundary aligns with the true shoreline. Points where the y-distance exceeds a threshold (bright: 30, timex: 10 pixels) are flagged as outliers and excluded from the final shoreline obtained from the median bottom boundary of the SAM model. 


process_bright(self)
----------------------

The `process_bright` method processes images of the `bright` type and extracts shoreline data through several steps. This method is specifically designed to handle high-contrast images where the surf zone is well-defined.

**Workflow:**

- **Metadata Extraction:**
  - Extracts metadata (e.g., month, day, time, year, site, and camera) directly from the image file name.
  - These details are used to store results in the `ShorelineDatastore`.

- **Surfzone Point Identification:**
  - Locates random points in the surfzone region by:
    - Processing the image to find the largest connected component above an Otsu threshold (typically representing the surf zone).
    - Selecting random points from this region to initialize the SAM segmentation model (minimum 5 points or every 200 pixels).
  - These surfzone points anchor the segmentation process.

- **Bottom Boundary Detection:**
  - Attempts to extract the bottom boundary of the surf zone three times using the SAM (Segment Anything Model) for segmentation:
    - Surfzone points are used to predict a segmentation mask for the surf zone.
    - The bottom boundary is extracted from the mask as the maximum y-value for each x-coordinate.
    - Each x-coordinate has only one y-value, which may cause issues in along-shore camera angles with edge waves.
  - Repeats the process three times to reduce noise, calculating the median of the three boundaries as the final bottom boundary.

- **Watershed Segmentation:**
  - The median bottom boundary is used as the boundary between ocean and sand.
  - Random points above and below this boundary are inputs to the watershed segmentation algorithm.
  - The watershed algorithm refines the shoreline boundary by separating water from sand.

- **Metrics Calculation:**
  - Calculates:
    - **Y-Distance** between the watershed boundary and the SAM-extracted bottom boundary.
    - **Root Mean Squared Error (RMSE)** for segmentation accuracy.
  - Flags outliers where the y-distance exceeds 30 pixels, excluding them from the final shoreline.

- **Visualization:**
  - If `make_intermediate_plots` is enabled:
    - Visualizations are generated, including:
      - The original image overlaid with detected boundaries.
      - Intermediate segmentation masks and watershed results.
    - Final visualizations are saved in the `shoreline_plots` directory.

- **Data Storage:**
  - Stores the following in the `ShorelineDatastore`:
    - Final shoreline coordinates.
    - Bottom boundary and watershed segmentation results.
    - Computed metrics (e.g., RMSE, y-distance).

**Suitability:**
This method is ideal for `bright` images due to their higher contrast and well-defined boundaries, enabling accurate segmentation and analysis.

process_timex(self)
---------------------

The `process_timex` method processes images of the `timex` type, which are typically long-exposure images showing time-averaged wave patterns. This method leverages results from previously processed `bright` images to assist in segmentation.

**Workflow:**

- **Metadata Extraction:**
  - Extracts metadata (e.g., site, camera, date, and time) from the image file name.
  - Metadata is used to retrieve previously processed results and store new outputs.

- **Integration with `bright` Results:**
  - Retrieves shoreline coordinates from a corresponding `bright` image in the `ShorelineDatastore`.
  - These coordinates serve as the initial boundary for segmentation in the `timex` image.

- **Bottom Boundary Detection:**
  - Generates random points slightly above the retrieved shoreline coordinates to mark the surf zone in the `timex` image.
  - Performs SAM segmentation three times to extract bottom boundaries using these points.
  - Computes the median bottom boundary for stability.

- **Watershed Segmentation:**
  - Refines the median bottom boundary using the watershed segmentation algorithm to separate the shoreline from other regions.

- **Metrics Calculation:**
  - Calculates:
    - **Y-Distance** between the watershed boundary and the SAM-generated boundary.
    - **RMSE** for segmentation accuracy.
  - Applies a stricter threshold (10 pixels for y-distance) to filter outliers compared to `_process_bright`.

- **Visualization:**
  - If `make_intermediate_plots` is enabled:
    - Generates plots similar to `_process_bright`.
    - Includes the retrieved `bright` shoreline for comparison.

- **Data Storage:**
  - Stores the following in the `ShorelineDatastore`:
    - Shoreline coordinates.
    - Segmentation results.
    - Computed metrics (e.g., RMSE, y-distance).

**Suitability:**
This method is particularly effective for `timex` images, which often lack sharp contrasts. Integrating results from `bright` images ensures accurate shoreline detection.

