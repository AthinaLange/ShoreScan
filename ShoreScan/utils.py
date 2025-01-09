"""
utils Module
==========================
This module provides functions helpful for ShoreScan

Functions:
    - create_video_from_images
    - check_processed_images
    - removed_processed_images
    - process_images
"""

import cv2

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
                                            output_folder = "shoreline_output",
                                        )
                                    except Exception as e:
                                        print(f"Error processing {img_path}: {e}")

