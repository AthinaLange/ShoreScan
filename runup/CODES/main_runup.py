# ShoreScan
import argparse
from utils_play import *


def main(configPath = None):
    """
    Main function to execute the ShoreScan timestack workflow.
    - prep_ras_images()
        - Loads user-specified image data.
        - Filters images based on various criteria.
        - Split .tiff based on transects.
        - Embed exif metadata.
    - extract_runup()
        - flip images and save as grayscale
        - run segformer : SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel
        - extract runup lines from .npz file based off user runup/rundown parameters. Default 0, -1.5.
    - get_runup()
        - make image datastore with matching runup files and pull metadata
        - get x,y,z coordinates from DEM
        - pull hRunup and vRunup and save in netCDF files

    Requirements:
        - YAML with IO/EO
        - JSON with Site_c#
        - folder with all .tiff timestacks
        - camera_settings.json - info for .pix file
        - c1_timestacks_YYYYMMDD.pix - with location defined in camera_settings JSON file
        - c2_timestacks_YYYYMMDD.pix - with location defined in camera_settings JSON file
        - site_setting.json - info for netCDF

    Can input everything through config.json
    {
        "rawDir": "/path/to/raw/images",
        "timestackDir": "/path/to/timestacks",
        "camera_settingsDir": "/path/to/camera_settings.json",
        "grayscaleDir": "/path/to/grayscale",
        "overlayDir": "/path/to/overlays",
        "runup_val": 0.0,
        "rundown_val": -1.5,
        "segformerWeightsDir": "/path/to/segformer/weights",
        "model": "SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5",
        "segformerCodeDir": "/path/to/segformer/code",
        "site_settingsDir": "/path/to/site_settings.json",
        "split_tiff": false
    }

    Returns:
        None
    """

    #runup_folders = {
    #    "overlay_1_n3": "Runup = 1, Rundown = -3",
    #    "overlay_0p5_n2": "Runup = 0.5, Rundown = -2",
    #    "overlay_0_n1p5": "Runup = 0, Rundown = -1.5"
    #}
    #timestacks_folder = os.path.join(askdirectory(title="Data folder."))
    #timestacks_folder = os.path.join(timestacks_folder, 'timestacks')
    #plot_multiple_runup(timestacks_folder, runup_folders)

    config = None
    if configPath:
        with open(configPath, "r") as file:
            config = json.load(file)

    # Run functions with config (if loaded) or manual input
    timestackDir = prep_ras_images(config = config) if config else prep_ras_images()
    extract_runup(config = config) if config else extract_runup(timestackDir = timestackDir)
    get_runup(config = config) if config else get_runup(timestackDir = timestackDir)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ShoreScan timestack workflow.")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file", default=None)
    
    args = parser.parse_args()
    main(configPath=args.config)
