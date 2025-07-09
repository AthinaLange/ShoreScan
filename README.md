# ShoreScan 🌊

A Python-based tool for advanced image processing of ARGUS-style camera output, featuring automated image rectification and wave runup extraction from timestacks.

##  What is ShoreScan?

ShoreScan processes coastal camera imagery to extract valuable oceanographic data including:
- Automated image rectification
- Wave runup measurements
- Shoreline position tracking  
- Bathymetry inversion

Perfect for coastal researchers, marine scientists, and anyone working with coastal monitoring systems.


## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda package manager
-TODO add download Segment-anything & gym is missing from the project

## Installation

<details>
<summary><strong>Method 1: Using Environment File (Recommended)</strong></summary>
- TODO - ensure this works

```bash
    # Clone the repository
    git clone https://github.com/athinalange/ShoreScan.git
    cd ShoreScan/runup
    
    # Create environment
    conda env create --name shorescan -f shorescan_initial_config.yml
    conda activate shorescan
    
    # Install PyTorch with CUDA support
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Verify installation
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    python -c "from transformers import TFSegformerForSemanticSegmentation"
    
    # Install Segment Anything
    cd CODES/segment-anything-main
    pip install -e .
    cd ../..
```

</details>

<details>
<summary><strong>Method 2: Full Installation on WSL2 (Ubuntu 24.04.1)</strong></summary>

```bash
    # Start from segmentation-gym install
    conda env create --name shorescan -f gym.yml
    conda activate shorescan
    
    # Install system dependencies
    sudo apt install libimage-exiftool-perl
    
    # Install conda packages
    conda install xarray netcdf4 numpy=1.24.* plotly scikit-learn ipykernel opencv piexif
    
    # Install pip packages
    pip install utm segment-anything pyexiftool onnxruntime onnx ipython rioxarray geopy
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Test installation
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    python -c "from transformers import TFSegformerForSemanticSegmentation"
    
    # Install Segment Anything
    cd CODES/segment-anything-main
    pip install -e .
    cd ../..
```

</details>

<details>
<summary><strong>Method 3: macOS Installation</strong></summary>

```bash
    conda env create --name shorescan -f shorescan_initial_config_mac.yml
    conda activate shorescan
    conda install xarray netcdf4 numpy plotly scikit-learn ipykernel opencv piexif -y
    pip3 install torch torchvision torchaudio
    
    # Install Segment Anything
    cd CODES/segment-anything-main
    pip install -e .
    cd ../..
```

</details>

<details>
<summary><strong>Additional Setup</strong></summary>

#### Required Model Download
Download `sam_vit_h_4b8939.pth` from [Segment Anything](https://github.com/facebookresearch/segment-anything) and place it in `segment-anything-main/`.

#### PyTorch 2.6 Fix
Update `segment-anything-main/segment-anything/build_sam.py` line 105:
```text
    state_dict = torch.load(f, weights_only=False)
```

</details>

## Usage

### Basic Usage

```bash
    # Run the main processing notebook
    jupyter notebook CODES/CoastCam_processing.ipynb
```


### Input Requirements

**Required:**

- ARGUS-style images:
    - Timestack images in one of two formats:
      - `.tiff` files containing concatenated timestacks
      - `.png/.jpg` files of individual timestacks (dimensions: time × length(U,V) coordinates)
    - Oblique images, e.g. snap, timex, var, bright, dark
- YAML files with IO/EO/metadata
- JSON configuration files
    - site specific information
    - camera specific information
    - products to generate coordinates
    - metadata information
- U,V coordinate files for timestack images

**Optional:**
- `config.json` for automated directory and variable definitions
- DEM files for elevation-based runup projection

## Configuration

<details>
<summary><strong>📝 Main Configuration Files</strong></summary>

#### `config.json` - Main Configuration
Provides the path to all relevant directories, files, thresholds and indices. 

We recommend having one `config.json` file per site.
```json
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
    "productsPath": "/path/to/products/dictionary/camera_products.json",
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
    "site_id": "TWL site_id as int",
    "station_id": "NOAA tide gauge station id",
    "NOAA_datum": "NAVD",
    "tide": "tide_level"
}
```

</details>

<details>
<summary><strong>📷 Camera Configuration</strong></summary>

#### `camera_settings.json` - Camera Configuration
Specifies which U,V coordinate file (.pix) is used for which camera during which time period.

Multiple files can be given in the `coordinate_files`.
```json
{
  "SITE_ID": {
    "CHANNEL_ID": {
      "reverse_flag": false,
      "coordinate_files": {
        "START_TIME|END_TIME": "file_path"
      }
    }
  }
}
```

**Fields:**
- `SITE_ID`: Site identifier (e.g., "CACO03", "CACO04")
- `CHANNEL_ID`: Channel identifier (e.g., "c1", "c2")
- `reverse_flag`: Whether to reverse pix coordinates (false = offshore to onshore)
- `coordinate_files`: Time range to file path mapping (ISO 8601 format)


</details>

<details>
<summary><strong>🏖️ Site Metadata</strong></summary>

#### `site_settings.json` - Site Metadata

```json
{
  "SITE_ID": {
    "siteName": "Full site name",
    "shortName": "Short identifier", 
    "siteInfo": {
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
    "sampling": {
      "sample_frequency": "Number",
      "collection_unit": "Unit of frequency (Hz, seconds, etc.)",
      "sample_period_length": "Number",
      "sample_period_unit": "Unit for sample period (s, min, etc.)",
      "freqLimits": ["Upper SS limit", "SS/IG transition limit", "Lower IG limit"]
    }
  }
}
```
**Key Fields:**
- `sample_frequency`: Sampling frequency (e.g., 2, 5)
- `sample_period_length`: Duration of each sampling period (e.g., 600) 
- `freqLimits`: [Upper SS limit, SS/IG transition limit, Lower IG limit], (eg. [0.5, 0.05,0.04])

</details>

<details>
<summary><strong>📊 Grid Configuration</strong></summary>

#### `products.json` - Grid Configuration
Provides the information for the rectified grid.
```json
{
  "type": "Grid", "xTransect", "yTransect",
  "frameRate": "Sampling frequency in Hertz",
  "lat": "Latitude of origin",
  "lon": "Longitude of origin",
  "east": "Eastings of origin (UTM)",
  "north": "Northings of origin (UTM)",
  "zone": "UTM zone",
  "angle": "shorenormal angle, CW from North",
  "xlim": "[onshore limit of grid relative to origin, offshore limit of grid relative to origin] (positive is offshore, e.g. [0, 200])",
  "ylim": "[rightside limit of grid relative to origin, leftside limit of grid relative to origin] (right is negative, looking offshore, e.g. [-100, 300])",
  "dx": "cross-shore grid spacing",
  "dy": "along-shore grid spacing",
  "x": "cross-shore location (e.g. 100 or null)",
  "y": "along-shore location (e.g. 100 or null)",
  "z": "vertical elevation of transect (e.g. 2 or null)",
  "tide": "tide level relative to tide datum (offsets grid to this elevation)",
  "lim_flag": "flag specifiying if xlim/ylim definied relative to geographical or local coordinates (0 = local, 1 = UTM)"
}
```
If east, north, and zone are not relevant, will default to lat/lon.

</details>

## File Structure

<details>
<summary><strong>Complete Project Structure</strong></summary>

```
ShoreScan/
├── docs/                                            # Documentation files
├── runup/                                           # Main application code
│   ├── CODES/                                       # Core source code modules
│   │   ├── segment-anything-main/                   # Segment Anything model implementation
│   │   ├── ImageHandler.py                          # Image processing and manipulation utilities
│   │   ├── seg_images_in_folder.py                  # Batch image segmentation functionality
│   │   ├── segformer.py                             # SegFormer model implementation
│   │   ├── utils_CIRN.py                            # CIRN (Coastal Imaging Research Network) utilities
│   │   ├── utils_exif.py                            # EXIF data extraction and processing
│   │   ├── utils_runup.py                           # Wave runup calculation utilities
│   │   ├── utils_segformer.py                       # SegFormer-specific utility functions
│   │   └── utils_shoreline.py                       # Shoreline detection and analysis tools
│   ├── DATA/                                        # Data storage and configuration
│   │   └── DATA/                                    # Nested data directory
│   │       ├── images/                              # Image datasets
│   │       ├── CAC003_c1_timestack_20240920.pix     # Timestack camera 1 U,V coordinates 
│   │       └── CAC003_c2_timestack_20240920.pix     # Timestack camera 2 U,V coordinates 
│   ├── JSON/                                        # JSON configuration files
│   ├── segmentation_gym/                            # Segmentation model training data
│   │   ├── config/                                  # Configuration files for training
│   │   └── weights/                                 # Pre-trained model weights
│   ├── YAML/                                        # YAML configuration files
│   ├── CoastCam_processing.ipynb                    # Jupyter notebook for CoastCam data processing
│   ├── config.json                                  # Main configuration file
│   ├── config_example.json                          # Example configuration template
│   ├── README.txt                                   # Basic project information
│   ├── shorescan.yml                                # Main YAML configuration
│   └── shorescan_initial_config.yml                 # Initial setup configuration
├── LICENSE                                          # Project license information
├── README.md                                        # Main project documentation
├── ShoreScan.pdf                                    # Project documentation (PDF format)
└── shorescan_readme.md                              # Additional readme file
```
</details>

## Examples - WIP

<details>
<summary><strong> Basic Usage Examples</strong></summary>


### Processing a Single Site - WIP
```python
from CODES.ImageHandler import ImageHandler

# Initialize with configuration
handler = ImageHandler(config_path="config.json")

# Process images for a specific site and date
handler.process_site("CACO03", "20240920")
```

### Batch Processing - WIP
```python
# Process all sites in date range
sites = ["CACO03", "CACO04"]
dates = ["20240920", "20240921", "20240922"]

for site in sites:
    for date in dates:
        handler.process_site(site, date)
```

</details>

## Troubleshooting

<details>
<summary><strong>Common Issues & Solutions</strong></summary>

### GPU Not Detected
```bash
    # Check CUDA installation
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available())"
```

### Segment Anything Import Error
```bash
    # Reinstall with proper PyTorch version
    pip uninstall torch torchvision torchaudio
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Missing Dependencies
```bash
    # Install missing system packages (Ubuntu/Debian)
    sudo apt update
    sudo apt install libimage-exiftool-perl
```

</details>




## Acknowledgments

- [Segment Anything](https://github.com/facebookresearch/segment-anything) for segmentation capabilities
- [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) for model training tools
- ARGUS coastal monitoring community
