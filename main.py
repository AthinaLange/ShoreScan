# ShoreScan

from c_ImageDatastore import ImageDatastore
from c_ShorelineDatastore import ShorelineDatastore
from c_ShorelineWorkflow import ShorelineWorkflow

#from create_video_from_images import create_video_from_images

import os
import matplotlib.pyplot as plt


def check_processed_images(image_metadata, pt_dir, output_dir):
    """
    Check if an image has been processed by verifying existence in pt_dir and output_dir.

    Args:
    - image_metadata (dict): Metadata of the image to check.
    - pt_dir (str): Directory where shoreline point files are stored.
    - output_dir (str): Directory where shoreline output files are stored.

    Returns:
    - bool: True if processed, False otherwise.
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
    Remove processed images from the datastore by checking if they exist in the output directories.

    Args:
    - datastore (ImageDatastore): The datastore containing image metadata.
    - pt_dir (str): Directory for shoreline point files.
    - output_dir (str): Directory for shoreline output files.
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
    Process images of a given type using the ShorelineWorkflow class.

    Args:
    - datastore (ImageDatastore): The datastore containing image metadata.
    - img_type (str): The image type to process (e.g., 'bright', 'timex').
    - shoreline_datastore (ShorelineDatastore): Datastore to store shoreline analysis results.
    - make_intermediate_plots (bool): Whether to generate intermediate plots during processing.
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


# Initialize the datastore
#datastore = ImageDatastore(root_folder='data')
datastore = ImageDatastore()

# Load the images into the datastore
datastore.load_images()
# List all images loaded into datastore
#datastore.list_images()

# Ask user for processing options
user_choice = input("Do you want to process all images or only new ones? (all/new): ").strip().lower()
if user_choice == 'new':
    pt_dir = 'shoreline_plots'
    output_dir = 'shoreline_output'
    remove_processed_images(datastore, pt_dir, output_dir)

# Ask if intermediate plots should be generated
user_input = input("Do you want to run with intermediate plots? (yes/no): ").strip().lower()
make_intermediate_plots = user_input in ['yes', 'y']

# Filter out images based on various criteria
datastore.filter_black_images()
datastore.filter_white_images()
datastore.filter_sun_glare(csv_file = 'camera_sites.csv')
# Filter for blurry images with thresholds (fog and rain)
thresholds = {
    'snap' : 20,
    'timex' : 15,
    'bright' : 35,
    'dark' : 20,
    'var' : 30
}
datastore.filter_blurry_images(blur_thresholds = thresholds)
del thresholds

# Display image stats
datastore.image_stats()

# Initialize shoreline datastore for storing results
shoreline_datastore = ShorelineDatastore()

# Process "bright" images
process_images(
    datastore = datastore,
    img_type = 'bright',
    shoreline_datastore = shoreline_datastore,
    make_intermediate_plots = make_intermediate_plots,
)

# Process "timex" images
process_images(
    datastore = datastore,
    img_type = 'timex',
    shoreline_datastore = shoreline_datastore,
    make_intermediate_plots = make_intermediate_plots,
)

"""
# Optional bits of example code
#datastore.copy_images_to_folder('all_images', hierarchical=True)
#datastore.load_camera_site_info(csv_file='camera_sites.csv')
#print(datastore.camera_sites['CACO03']['lat'])
#image_types = list(datastore.images['CACO03']['c2']['2024']['Sep']['25']['14_00_00'])
#plt.imshow(cv2.imread(datastore.images['CACO03']['c2']['2024']['Aug']['22']['14_00_00']['timex'][0]['path']))
#datastore.plot_image(datastore.images['CACO03']['c2']['2024']['Aug']['22']['14_00_00']['timex'][0])

# Setup different cases
# c1_snap = datastore.get_image_metadata_by_type(['snap'], camera='c1')
# c2_snap = datastore.get_image_metadata_by_type(['snap'], camera='c2')

# c1_timex = datastore.get_image_metadata_by_type(['timex'], camera='c1')
# c2_timex = datastore.get_image_metadata_by_type(['timex'], camera='c2')

# c1_bright = datastore.get_image_metadata_by_type(['bright'], camera='c1')
# c2_bright = datastore.get_image_metadata_by_type(['bright'], camera='c2')

# c1_dark = datastore.get_image_metadata_by_type(['dark'], camera='c1')
# c2_dark = datastore.get_image_metadata_by_type(['dark'], camera='c2')

# create_video_from_images(datastore, video_name = 'c1_bright.mp4', frame_rate=5, image_type='bright', camera='c1')
"""
