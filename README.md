# ShoreScan_runup

Python-based tool for image processing of output from ARGUS-style cameras, including image rectification and runup extraction from timestacks

See more documentation [here](https://athinalange.github.io/ShoreScan/)

---
## Installation
Use shorescan.yml file 

Download [Segment-Anything](https://github.com/facebookresearch/segment-anything) and install in conda environment.
Download [segmentation-gym](https://github.com/Doodleverse/segmentation_gym) and install in conda environment.

### Install through .yml file
```
conda env create --name shorescan -f shorescan_initial_config.yml
conda activate shorescan
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "from transformers import TFSegformerForSemanticSegmentation"

cd CODES/segment-anything-main; pip install -e .;cd ..;cd ..
```

### Full install on WSL2 (Ubuntu 24.04.1)
Start from segmentation-gym install with gym.yml 
```
conda env create --name shorescan -f gym.yml
conda activate shorescan
sudo apt install libimage-exiftool-perl
conda install xarray netcdf4 numpy=1.24.* plotly scikit-learn ipykernel opencv piexif
pip install utm segment-anything pyexiftool onnxruntime onnx ipython rioxarray geopy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "from transformers import TFSegformerForSemanticSegmentation"

cd CODES/segment-anything-main; pip install -e .;cd ..;cd ..
```
### Additional things

Update to segment-anything-main/segment-anything/build_sam.py line 105
pytorch version 2.6 requires: ``` state_dict = torch.load(f, weights_only=False) ```

Please download model: sam_vit_h_4b8939.pth from segment-anything and put it in segment-anything-main

## Run
User prompting:
```
CoastCam_processing.ipynb
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


camera_settings.json:													<br/>
{															<br/>
&emsp;"SITE_ID": {  # Each site (e.g., "CACO03", "CACO04") is a key							<br/>
&emsp;&emsp;"CHANNEL_ID": {  # Each channel (e.g., "c1", "c2") under the site						<br/>
&emsp;&emsp;&emsp;"reverse_flag": bool,  # Indicates if pix coordinates should be reversed (false: offshore to onshore)	<br/>
&emsp;&emsp;&emsp;"coordinate_files": {  # Maps time ranges to file paths						<br/>
&emsp;&emsp;&emsp;"START_TIME|END_TIME": "file_path"  # Time range (ISO 8601 format) mapped to a file			<br/>
&emsp;&emsp;&emsp;}													<br/>
&emsp;&emsp;}														<br/>
&emsp;}															<br/>
}

site_settings.json:													<br/>
{															<br/>
&emsp; "SITE_ID": {  // Each site (e.g., "CACO03", "SITEB") is a key containing site-specific information		<br/>
&emsp;&emsp;"siteName": "Full site name",  // Descriptive name of the site						<br/>
&emsp;&emsp;"shortName": "Short identifier",  // Abbreviated site name							<br/>
&emsp;&emsp;"siteInfo": {  // Metadata related to the site								<br/>
&emsp;&emsp;&emsp;"siteLocation": "Geographical location of the site",							<br/>
&emsp;&emsp;&emsp;"dataOrigin": "Organization responsible for the data",						<br/>
&emsp;&emsp;&emsp;"camMake": "Camera manufacturer",									<br/>
&emsp;&emsp;&emsp;"camModel": "Camera model",										<br/>
&emsp;&emsp;&emsp;"camLens": "Lens specifications",									<br/>
&emsp;&emsp;&emsp;"timezone": "Local timezone",										<br/>
&emsp;&emsp;&emsp;"utmZone": "UTM coordinate zone",									<br/>
&emsp;&emsp;&emsp;"verticalDatum": "Vertical reference system",								<br/>
&emsp;&emsp;&emsp;"verticalDatum_description": "Description of the vertical datum",					<br/>
&emsp;&emsp;&emsp;"references": "Citation or source reference for the data",						<br/>
&emsp;&emsp;&emsp;"contributors": "Names of individuals who contributed to data collection",				<br/>
&emsp;&emsp;&emsp;"metadata_link": "URL to metadata and dataset information"						<br/>
&emsp;&emsp;},														<br/>
&emsp;&emsp;"sampling": {  // Sampling configuration for data collection						<br/>
&emsp;&emsp;&emsp;"sample_frequency": Number,  // Sampling frequency (e.g., 2, 5)					<br/>
&emsp;&emsp;&emsp;"collection_unit": "Unit of frequency (Hz, seconds, etc.)",						<br/>
&emsp;&emsp;&emsp;"sample_period_length": Number,  // Duration of each sampling period					<br/>
&emsp;&emsp;&emsp;"sample_period_unit": "Unit for sample period (s, min, etc.)",					<br/>
&emsp;&emsp;&emsp;"freqLimits": [Upper SS limit,SS/IG transition limit,Lower IG limit]					<br/>
&emsp;&emsp;}														<br/>
&emsp;}															<br/>
}															<br/>

products.json
{															<br/>
&emsp;"type": "Grid",													<br/>
&emsp;"frameRate": 2,													<br/>
&emsp;"lat": origin latitude,												<br/>
&emsp; "lon": origin longitude,												<br/>
&emsp;"east": utm eastings (priority over lat/lon),									<br/>
&emsp;"north": utm northings (priority over lat/lon), 									<br/>
&emsp;"zone": utm zone,													<br/>
&emsp;"angle": shorenormal angle from north,										<br/>
&emsp;"xlim": [0, 200],													<br/>
&emsp;"ylim": [ -100, 300],												<br/>
&emsp;"dx": cross-shore spacing in meters,										<br/>
&emsp;"dy": along-shore spacing in meters,										<br/>
&emsp;"x": null,													<br/>
&emsp;"y": null,													<br/>
&emsp;"z": null,													<br/>
&emsp;"tide": 0,													<br/>
&emsp;"lim_flag": 0													<br/>
}

config.json:
 {															<br/>
&emsp;"imageDir": "path/to/images",											<br/>
&emsp;"jsonDir": "/path/to/json/folder",										<br/>
&emsp;"yamlDir": "path/to/yaml/folder",											<br/>
&emsp;"grayscaleDir": "/path/to/grayscale",										<br/>
&emsp;"runupDir": "/path/to/runup/files",										<br/>
&emsp;"videoDir": "path/to/video/folder",										<br/>
&emsp;"merged_rectifiedDir": "path/to/merged/and/rectified/images",							<br/>
&emsp;"pixsaveDir": "path/to/folder/to/save/pix",									<br/>
&emsp;"netcdfDir": "path/to/netcdf",											<br/>
&emsp;"camera_settingsPath": "/path/to/camera_settings.json",								<br/>
&emsp;"site_settingsPath": "/path/to/site_settings.json",								<br/>
&emsp;"productsPath": "path/to/products/dictionary/CACO03_products.json",						<br/>
&emsp;"segformerWeightsDir": "/path/to/segformer/weights",								<br/>
&emsp;"model": "SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5",						<br/>
&emsp;"segformerCodeDir": "/path/to/segformer/code",									<br/>
&emsp;"split_tiff": false,												<br/>
&emsp;"runup_val": 0.0,													<br/>
&emsp;"rundown_val": -1.5,												<br/>
&emsp;"thresholds": {													<br/>
&emsp;&emsp;"snap" : 20,												<br/>
&emsp;&emsp;"timex" : 15,												<br/>
&emsp;&emsp;"bright" : 35,												<br/>
&emsp;&emsp;"dark" : 20,												<br/>
&emsp;&emsp;"var" : 30													<br/>
&emsp;}															<br/>
}


Example Folder Structure

ShoreScan/ 														<br/>
│── CODES   														<br/>          
│&emsp;├── ImageHandler.py        				# Class to process images 				<br/>
│&emsp;├── CoastCam_processing.ipynb                		# Main script  						<br/>
│&emsp;├── RunUpTimeseriesFunctions_CHI.py               	# Code to compute runup statistics 			<br/>
│&emsp;├── seg_images_in_folder.py                		# Script to segment timestacks 				<br/>
│&emsp;├── segformer.py                				# segformer definition 					<br/>
│&emsp;├── utils_exif.py   					# utils functions          				<br/>
│&emsp;├── utils_segformer.py   				# utils functions          				<br/> 
│&emsp;├── utils_CIRN.py   					# utils functions          				<br/>
│&emsp;├── utils_runup.py   					# utils functions           				<br/>     
│── DATA                               											<br/>
│&emsp;├── DATA/ 													<br/>
│&emsp;&emsp;├── DEM/ 													<br/>
│&emsp;&emsp;&emsp;├── something DEM				# Elevation for runup projection 			<br/>
│&emsp;&emsp;├── images/	 											<br/>
│&emsp;&emsp;&emsp;├── *.tiff					# Raw .tiff images from camera				<br/>
│&emsp;&emsp;&emsp;├── *.jpg					# ARGUS image products					<br/>
│&emsp;&emsp;&emsp;├── *.png					# Split transects					<br/>
│&emsp;&emsp;├── CACO03_c1_timestack_20240920.pix		# c1 U,V coordinates for camera				<br/>
│&emsp;&emsp;├── CACO03_c2_timestack_20240920.pix		# c2 U,V coordinates for camera				<br/>
│&emsp;├── JSON/  													<br/>
│&emsp;&emsp;├── CACO03EXIF_c1.json				# c1 site info - for exif				<br/>
│&emsp;&emsp;├── CACO03EXIF_c1.json				# c2 site info - for exif				<br/>
│&emsp;&emsp;├── camera_settings.json				# Info for .pix locations				<br/>
│&emsp;&emsp;├── site_settings.json				# Info for netCDF					<br/>		
│&emsp;&emsp;├── CACO03_products.json				# Info for products (grid info)				<br/>		
│&emsp;├── segmentation_gym/												<br/>
│&emsp;&emsp;├── config/												<br/>
│&emsp;&emsp;&emsp;├── SegFormer_Madeira_Duck_equal.json				# segfomer model		<br/>
│&emsp;&emsp;&emsp;├── SegFormer_Madeira_Duck_equal_finetune_Waiakane.json		# segformer model		<br/>
│&emsp;&emsp;├── weights/    												<br/>
│&emsp;&emsp;&emsp;├── SegFormer_Madeira_Duck_equal_fullmodel.h5			# segformer model weights	<br/>
│&emsp;&emsp;&emsp;├── SegFormer_Madeira_Duck_equal_finetune_Waiakane_fullmodel.h5  	# segformer model weights       <br/>
│&emsp;├── YAML/           												<br/>
│&emsp;&emsp;├── CACO03_c1_20240801_IO.yaml			# c1 IO yaml						<br/>
│&emsp;&emsp;├── CACO03_c2_20240801_IO.yaml			# c2 IO yaml						<br/>
│&emsp;&emsp;├── CACO03_c1_20241023_EO.yaml			# c1 EO yaml						<br/>
│&emsp;&emsp;├── CACO03_c2_20241023_EO.yaml			# c2 EO yaml						<br/>
│&emsp;&emsp;├── CACO03_c1_20241023_metadata.yaml		# c1 metadata yaml					<br/>
│&emsp;&emsp;├── CACO03_c2_20241023_metadata.yaml		# c2 metadata yaml					<br/>
├── config.json                					# Configuration file with paths and parameters		<br/>
│── README.md                					# Project documentation					<br/>
