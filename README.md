# CSCI-2370 3D Transmission Grid Visualization Project

Hello, and thank you for participating in this project!

## Getting started
1. Ensure you have Python 3.11 available on your machine. If not, you can find instructions here: https://www.python.org/downloads/release/python-3110/
2. Ensure you have conda installed on your machine. If not, you can find instructions here (we suggest miniconda): https://www.anaconda.com/docs/getting-started/miniconda/main
3. Navigate to the `src/3dtgp` subdirectory; i.e. `cd src/3dtgp`
4. Initailize the enviornment with the following commands:
```sh
conda env create -f enviornment.yml
```
and then install the rasterio package from pypi:
```sh
pip install rasterio
```
5. You should be able to start the application! Run the following commands:
```sh
conda activate 3dtgp
python3 3d.py
```
