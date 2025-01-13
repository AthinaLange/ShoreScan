# ShoreScan

Python-based tool for a semi-automated shoreline detection workflow from standard ARGUS bright and timex images.

See more documentation [here](https://athinalange.github.io/ShoreScan/)

---
## Installation
Use shorescan.yml file 

Download [Segment-Anything](https://github.com/facebookresearch/segment-anything) and install in conda environment.

## Run
```
python main.py
```
- Navigate to 'data_sample'
- 'all'
- 'no'

After running the workflow through once, you will want to check the plot results in 'shoreline_plots'. If they are incorrect, you can delete the images or the .txt file and rerun python main.py and select 'new' to rerun the one that have not been fully processed and repeat confirming that the determined shoreline is correct. 
