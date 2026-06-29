# CloudCompare Script

## 3D Volume Analysis for Archaeological Stratigraphic Units

Part of the **[Tharros Archaeological Research Project (TARP)](https://air.ht.lu.se/s/tharros/page/home)** — an automated pipeline for converting 3D PLY models into volumetric measurements of archaeological Stratigraphic Units (SUs). The workflow combines automated preprocessing, manual human-in-the-loop refinement in CloudCompare, and automated post-processing to generate accurate 3D volumes.

## Overview

The pipeline processes paired top and bottom 3D models (PLY files) representing archaeological layers to compute the volume of material between them. This is achieved through a two-stage process:

1. **Pre-snip Script** (`pre_snip_script.py`): Automated preprocessing that computes distances between top and bottom point clouds
2. **Manual Snipping**: Human-guided refinement in CloudCompare to isolate regions of interest and remove outliers
3. **Post-snip Script** (`post_snip_script.py`): Automated mesh generation and volume calculation using Poisson reconstruction

## Features

- **Automated Distance Computation**: Calculates bidirectional distances between paired 3D point clouds
- **Flexible Input Configuration**: JSON-based configuration for specifying top/bottom model pairs
- **Volume Measurements**: Generates both 3D mesh volumes and 2.5D projected volumes
- **Quality Control**: Identifies potential mesh issues and provides warnings
- **Output Tracking**: Maintains volume measurements in structured text format

## Project Structure

```
cloudcomparescript/
├── pre_snip_script.py          # Preprocessing script for distance computation
├── post_snip_script.py         # Post-processing script for mesh generation
├── input.json                  # Configuration file for model pairs
├── volume_measures.txt         # Output volume measurements
├── Data/                       # Working directory
│   ├── Final_Volumes/          # Final SU volume meshes (OBJ files)
│   ├── Final_Volume_Tops/      # Top surface meshes (OBJ files)
│   └── Pgram_Job_*/            # Intermediate processing folders
└── ~/TARP/                     # Input PLY files directory
```

## Prerequisites

### CloudCompare with Python Support
- **macOS**: Download from [CloudCompare macOS binaries](https://www.simulation.openfields.fr/index.php/cloudcompare-downloads/7-cloudcompare-20250314-2-14-alpha-with-python-plugin)
- **Linux**: Follow [Linux setup instructions](https://github.com/CloudCompare/CloudComPy/blob/master/doc/UseLinuxCondaBinary.md)
- **Documentation**: [CloudCompare Python Runtime](https://tmontaigu.github.io/CloudCompare-PythonRuntime/index.html)

### System Requirements
- Python 3.10
- Conda package manager
- Sufficient RAM for 3D processing (recommended: 8GB+)

## Installation

### 1. Set up Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh  
conda create --name CloudComPy310 python=3.10
conda activate CloudComPy310
conda config --add channels conda-forge
conda config --set channel_priority flexible
```

### 2. Install Dependencies

```bash
conda install -y boost cgal cmake draco "ffmpeg=6.1" gdal jupyterlab laszip \
matplotlib "mysql=8" notebook numpy opencv "openssl=3.1" pcl pdal psutil \
pybind11 quaternion "qhull=2020.2" "qt=5.15.8" scipy sphinx_rtd_theme \
spyder tbb tbb-devel "xerces-c=3.2" xorg-libx11
```

### 3. Activate CloudComPy Environment

**macOS:**
```bash
cd ~/Desktop/CloudComPy310
source bin/condaCloud.zsh activate CloudComPy310
```

**Ubuntu:**
```bash
cd ~/CloudComPy/installConda/CloudComPy310/
. bin/condaCloud.sh activate CloudComPy310
```

### 4. Set up Directory Structure

```bash
mkdir -p Data/Final_Volumes
mkdir -p Data/Final_Volume_Tops
mkdir -p ~/TARP/
```

## Usage Workflow

### Step 1: Configure Input Models

1. Place your PLY files in `~/TARP/` with naming convention: `Pgram_Job_<job_number>_SU<su_numbers>.ply`
2. Edit `input.json` to specify top/bottom model pairs:

```json
[
  {
    "top": "721",
    "bottom": "728"
  },
  {
    "top": "714",  
    "bottom": "722"
  }
]
```

### Step 2: Run Preprocessing

```bash
python pre_snip_script.py
```

This script will:
- Load and sample meshes from PLY files
- Compute bidirectional distances between paired models
- Generate point clouds with distance scalar fields
- Save intermediate files in `Data/` folders

### Step 3: Manual Refinement in CloudCompare

1. Open the generated BIN files in CloudCompare
2. Manually select and isolate the Stratigraphic Unit areas of interest
3. Remove outlier points and noise
4. Save cleaned files as: `Pgram_Job_<pgram#>_SU<su-numbers>_cleaned_<specific_su_number>.bin`
5. For top clouds with higher job numbers, append `_top` to filename

### Step 4: Generate Final Volumes

```bash
python post_snip_script.py
```

This script will:
- Process all cleaned BIN files
- Merge top and bottom point clouds
- Apply Poisson surface reconstruction
- Calculate 3D and 2.5D volumes
- Generate final OBJ mesh files
- Update `volume_measures.txt` with measurements

## Output Files

- **Volume Meshes**: `Data/Final_Volumes/SU_<su_number>_raw.obj`
- **Top Surface Meshes**: `Data/<pgram_folder>/SU_<su_number>_top_raw.obj`
- **Volume Measurements**: `volume_measures.txt` (tab-separated format)
- **Project Files**: CloudCompare project files for review

## Configuration

### JSON Configuration Format

The `input.json` file defines model pairs where:
- `"top"`: Job number of the upper archaeological layer
- `"bottom"`: Job number of the lower archaeological layer

### Distance Computation Parameters

The preprocessing script supports various parameters for distance computation:
- Octree subdivision level (default: 7)
- Maximum distance threshold  
- Local modeling method (QUADRIC, LS, TRI, NO_MODEL)
- Multi-threading options

### Volume Calculation

Volume measurements include:
- **3D Volume**: Computed from Poisson-reconstructed mesh (cubic centimeters)
- **2.5D Volume**: Projected volume between surfaces
- **Quality Warnings**: Alerts for potential mesh holes or issues

## Troubleshooting

### Ubuntu Build Issues
If experiencing issues with the CloudComPy build script, try disabling non-essential plugins:

```bash
-DPLUGIN_IO_QCORE:BOOL="1" \
-DPLUGIN_IO_QADDITIONAL:BOOL="1" \
-DPLUGIN_IO_QE57:BOOL="1" \
-DPLUGIN_GL_QEDL:BOOL="1" \
-DPLUGIN_GL_QSSAO:BOOL="1" \
-DPLUGIN_IO_QLAS:BOOL="1" \
-DPLUGIN_STANDARD_QPOISSON_RECON:BOOL="1"
```

### Memory Issues
- Reduce Poisson reconstruction depth parameter for large datasets
- Process models in smaller batches
- Ensure adequate system RAM

### Missing Files
- Verify PLY files exist in `~/TARP/` directory
- Check JSON configuration matches actual file names
- Ensure proper naming conventions are followed

## Future Enhancements

- **Incremental Processing**: Skip already processed SUs in `Data/Final_Volumes`
- **Improved Filtering**: Better automated density-based filtering for top surfaces
- **Batch Processing**: Enhanced handling of large model collections
- **Validation Tools**: Automated quality checking of generated volumes

## Contributing

This project is part of ongoing archaeological research. For contributions or questions, please follow standard academic collaboration practices and cite appropriately in any derived work.

## License

This project is developed for the Tharros Archaeological Research Project. Please respect archaeological data and follow appropriate usage guidelines.
