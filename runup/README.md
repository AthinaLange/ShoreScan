# ShoreScan_runup

Python-based tool for runup processing from timestack output from ARGUS-style cameras.

See more documentation [here](https://athinalange.github.io/ShoreScan/)

---
## Installation
Use exif_shorescan.yml file 

Download [Segment-Anything](https://github.com/facebookresearch/segment-anything) and install in conda environment.
Download [segmentation-gym](https://github.com/Doodleverse/segmentation_gym) and install in conda environment.

## Run
User prompting:
```
python main_runup.py
```
Config prompting:
```
python main_runup.py --config /path/to/config.json
```

NOTE: 'write_netCDF' doesn't fully work yet bc dist_uv_to_xyz and DEM stuff needs to be written

## Requirements
- .tiff images of concatenated timestacks from ARGUS-style cameras in a folder OR
- .png/.jpg images of individual timestacks from ARGUS-style cameras. Must be dimensions of time x length(U,V) coordinates
- YAML folder with IO/EO/metadata
- JSON folder with site info
- camera_settings.json - info about pix definitions
- site_settings.json - info for netCDF
- U,V.pix coordinates (supplied to ARGUS-style camera) - location given in camera_settings.json

Optional:
- config.json - definitions for all directories and variables - no user prompting


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
		"directories": {  // File path locations for different types of data
                	"jpgDir": "Path to JPG image files",
                	"netcdfDir": "Path to NetCDF files",
                	"runupDir": "Path to runup analysis data",
                	"topoDir": "Path to topographic data",
			"yamlDir": "Path to YAML files"
			},
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

config.json:
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
        "split_tiff": false,
	"jsonDir": "/path/to/json/files",
	"yamlDir": "/path/to/yaml/files"
}




Example Folder Structure

ShoreScan_runup/
│── CODES                   
│   ├── c_Datastore.py        												    # Class to define ImageDatastore
│   ├── main_runup.py                											# Main script 
│   ├── RunUpTimeseriesFunctions_CHI.py               							# Code to compute runup statistics
│   ├── seg_images_in_folder.py                								# Script to segment timestacks
│   ├── segformer.py                											# segformer definition
│   ├── utils_play.py   														# utils functions              
│── DATA                               
│   ├── DATA/
│   	├── DEM/
│   		├── something DEM													# Elevation for runup projection
│   	├── raw/	
│   		├── *.tiff															# Raw .tiff images from camera
│   	├── CACO03_c1_timestack_20240920.pix									# c1 U,V coordinates for camera
│   	├── CACO03_c2_timestack_20240920.pix									# c2 U,V coordinates for camera
│   ├── JSON/  
│   	├── CACO03EXIF_c1.json													# c1 site info - for exif
│   	├── CACO03EXIF_c1.json													# c2 site info - for exif
│   	├── camera_settings.json												# Info for .pix locations
│   	├── site_settings.json													# Info for netCDF							
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
│   ├── config.json                											# Configuration file with paths and parameters
│── README.md                													# Project documentation
