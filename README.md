# ShoreScan_runup

Python-based tool for image processing of output from ARGUS-style cameras, including image rectification and runup extraction from timestacks

See more documentation [here](https://athinalange.github.io/ShoreScan/)

---
## Installation
Use exif_shorescan.yml file 

Download [Segment-Anything](https://github.com/facebookresearch/segment-anything) and install in conda environment.
Download [segmentation-gym](https://github.com/Doodleverse/segmentation_gym) and install in conda environment.

### Full install on WSL2 (Ubuntu 24.04.1)
Start from 'conda create -n shorescan -f shorescan_initial_config.yml'
conda activate shorescan
#sudo apt install libimage-exiftool-perl
#conda install xarray netcdf4 numpy=1.24.* plotly scikit-learn ipykernel opencv piexif
#pip install utm segment-anything pyexiftool onnxruntime onnx
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "from transformers import TFSegformerForSemanticSegmentation"

cd segment-anything-main; pip install -e .

Update to segment-anything-main/segment-anything/build_sam.py line 105
pytorch version 2.6 requires: ``` state_dict = torch.load(f, weights_only=False) ```


## Run
User prompting:
```
CoastCam_processing.ipynb
```


## Requirements
- .png/.jpg images of images to be rectified, individual or concatinated timestacks from ARGUS-style cameras. Must be dimensions of time x length(U,V) coordinates
- config.json - definitions for all directories and variables
- YAML folder with IO/EO/metadata
- JSON folder with site info
- camera_settings.json - info about pix definitions
- site_settings.json - info for netCDF
- products.json - info for projection grid
- U,V .pix coordinates (supplied to ARGUS-style camera) - location given in camera_settings.json - can also be definied in code
- DEM.tif for projection onto surface



camera_settings.json:
{
	"SITE_ID": {  # Each site (e.g., "CACO03", "CACO04") is a key
		"CHANNEL_ID": {  # Each channel (e.g., "c1", "c2") under the site
			"reverse_flag": bool,  # Indicates if pix coordinates should be reversed (false: offshore to onshore)
			"coordinate_files": {  # Maps time ranges to file paths
			"START_TIME|END_TIME": "file_path"  # Time range (ISO 8601 format) mapped to a file
			}
		}
	}
}

site_settings.json:
{
        "SITE_ID": {  // Each site (e.g., "CACO03", "SITEB") is a key containing site-specific information
		"siteName": "Full site name",  // Descriptive name of the site
		"shortName": "Short identifier",  // Abbreviated site name
		"siteInfo": {  // Metadata related to the site
                	"siteLocation": "Geographical location of the site",
                	"dataOrigin": "Organization responsible for the data",
                	"camMake": "Camera manufacturer",
                	"camModel": "Camera model",
                	"camLens": "Lens specifications",
                	"timezone": "Local timezone",
                	"utmZone": "UTM coordinate zone",
                	"verticalDatum": "Vertical reference system",
                	"verticalDatum_description": "Description of the vertical datum",
                	"references": "Citation or source reference for the data",
                	"contributors": "Names of individuals who contributed to data collection",
                	"metadata_link": "URL to metadata and dataset information"
            	},
		"sampling": {  // Sampling configuration for data collection
                	"sample_frequency": Number,  // Sampling frequency (e.g., 2, 5)
                	"collection_unit": "Unit of frequency (Hz, seconds, etc.)",
                	"sample_period_length": Number,  // Duration of each sampling period
                	"sample_period_unit": "Unit for sample period (s, min, etc.)",
                	"freqLimits": [  // Frequency range limits (if applicable)
                    		Upper SS limit,
                    		SS/IG transition limit,
                    		Lower IG limit
                		]
		}
	}
}

products.json
{
        "type": "Grid",
        "frameRate": 2,
        "lat": origin latitude,
        "lon": origin longitude,
        "east": utm eastings (priority over lat/lon),
        "north": utm northings (priority over lat/lon), 
        "zone": utm zone,
        "angle": shorenormal angle from north,
        "xlim": [0, 200],
        "ylim": [ -100, 300],
        "dx": cross-shore spacing in meters,
        "dy": along-shore spacing in meters,
        "x": null,
        "y": null,
        "z": null,
        "tide": 0,
        "lim_flag": 0
    }

config.json:
{
    "imageDir": "/path/to/images",
    "jsonDir": "/path/to/json/folder",
    "yamlDir": "/path/to/yaml/folder",
    "grayscaleDir": "/path/to/grayscale",
    "runupDir": "/path/to/runup/files",
    "videoDir": "/path/to/movie/folder",
    "merged_rectifiedDir": "/path/to/merged/and/rectified/images",
    "pixsaveDir":"/path/to/folder/to/save/pix",
    "netcdfDir": "/path/to/netcdf",
    "shorelineDir": "/path/to/shoreline",
    "twlDir": "/path/to/twl_forecast",
    "camera_settingsPath": "/path/to/camera_settings.json",
    "site_settingsPath": "/path/to/site_settings.json",
    "productsPath": "/path/to/products/dictionary/CACO03_products.json",
    "demPath": "/path/to/dem.tif",
    "segformerWeightsDir": "/path/to/segformer/weights",
    "segformerModel": "SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5",
    "segformerCodeDir": "/path/to/segformer/code",
    "segmentAnythingDir": "/path/to/segment-anything-main/",
    "segmentAnythingModel": "sam_vit_h_4b8939.pth",
    "split_tiff": false,
    "runup_val": 0.0,
    "rundown_val": -1.5,
    "thresholds": {
        "snap" : 20,
        "timex" : 15,
        "bright" : 35,
        "dark" : 20,
        "var" : 30
    },
    "f_lims": [0.004, 0.04, 0.35],
    "twl_region": "TWL region_id as int",
    "site_id": "TWL site_id as int"
}



Example Folder Structure

ShoreScan/
├── CoastCam_processing.ipynb                									# Main script 
├── config.json                												# Configuration file with paths and parameters
│── CODES                   
│   ├── ImageHandler.py        												# Class to process images
│   ├── RunUpTimeseriesFunctions_CHI.py               							# Code to compute runup statistics
│   ├── seg_images_in_folder.py                								# Script to segment timestacks
│   ├── segformer.py                											# segformer definition
│   ├── utils_exif.py   														# utils functions          
│   ├── utils_segformer.py   													# utils functions          
│   ├── utils_CIRN.py   														# utils functions         
│   ├── utils_runup.py   														# utils functions               
│   ├── segment-anything-main													# segment anything folder
│── DATA                               
│   ├── DATA/
│   	├── DEM/
│   		├── something DEM													# Elevation for runup projection
│   	├── images/	
│   		├── *.tiff															# Raw .tiff images from camera
│   		├── *.jpg															# ARGUS image products
│   		├── *.png															# Split transects
│   	├── CACO03_c1_timestack_20240920.pix									# c1 U,V coordinates for camera
│   	├── CACO03_c2_timestack_20240920.pix									# c2 U,V coordinates for camera
│   ├── JSON/  
│   	├── CACO03EXIF_c1.json													# c1 site info - for exif
│   	├── CACO03EXIF_c1.json													# c2 site info - for exif
│   	├── camera_settings.json												# Info for .pix locations
│   	├── site_settings.json													# Info for netCDF
│   	├── CACO03_products.json												# Info for products (grid info)		
│   ├── segmentation_gym/
│   	├── config/
│   		├── SegFormer_Madeira_Duck_equal.json								# segfomer model
│   		├── SegFormer_Madeira_Duck_equal_finetune_Waiakane.json			# segformer model
│   	├── weights/    
│   		├── SegFormer_Madeira_Duck_equal_fullmodel.h5						# segformer model weights
│   		├── SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5  	# segformer model weights            
│   ├── YAML/           													
│   	├── CACO03_c1_20240801_IO.yaml											# c1 IO yaml
│   	├── CACO03_c2_20240801_IO.yaml											# c2 IO yaml
│   	├── CACO03_c1_20241023_EO.yaml											# c1 EO yaml
│   	├── CACO03_c2_20241023_EO.yaml											# c2 EO yaml
│   	├── CACO03_c1_20241023_metadata.yaml									# c1 metadata yaml
│   	├── CACO03_c2_20241023_metadata.yaml									# c2 metadata yaml
│── README.md                													# Project documentation
