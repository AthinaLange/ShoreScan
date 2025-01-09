# ShoreScan

from c_Datastore import ImageDatastore, ShorelineDatastore
from c_Workflow import Workflow
from utils import check_processed_images, remove_processed_images, process_images
#from utils import create_video_from_images

import os
import matplotlib.pyplot as plt

def main():
    """
    Main function to execute the ShoreScan workflow.

    - Loads user-specified image data.
    - Filters images based on various criteria.
    - Processes shoreline images based on user-defined options.
    """
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
        pt_dir = 'data/shoreline_plots'
        output_dir = 'data/shoreline_output'
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

if __name__ == "__main__":
    main()