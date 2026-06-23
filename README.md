# CloudCompare Script

## 3D Volume Analysis for Archaeological Stratigraphic Units

Part of the **[Tharros Archaeological Research Project (TARP)](https://air.ht.lu.se/s/tharros/page/home)** — an automated pipeline for converting 3D PLY models into volumetric measurements of archaeological Stratigraphic Units (SUs). The workflow combines automated preprocessing, automated snipping via image registration and point cloud analysis, and automated post-processing to generate accurate 3D volumes.

## Overview

The pipeline processes paired top and bottom 3D models (PLY files) representing archaeological layers to compute the volume of material between them.

1. **Pre-snip** (`pre_snip_script.py`): Loads PLY meshes, samples them to point clouds, computes bidirectional cloud-to-cloud (C2C) distances, and saves `.bin` files with distance scalar fields.
2. **Auto-snip** (`auto_snip_script.py`): Automatically isolates each SU's region using an annotation image, replacing the previous manual CloudCompare step.
3. **Post-snip** (`post_snip_script.py`): Merges top and bottom cropped clouds, runs Poisson surface reconstruction, and computes 3D and 2.5D volumes.

## Project Structure

```
cloudcomparescript/
├── pre_snip_script.py          # Step 1: distance computation
├── auto_snip_script.py         # Step 2: automated snipping
├── post_snip_script.py         # Step 3: mesh generation and volume calculation
├── run.sh                      # Wrapper to run scripts with correct CloudComPy env
├── example.json                # Global job pair configuration
├── example-17000.json          # Per-season config with annotation filenames
├── SU_*_annotation.png         # Annotation images (one per SU)
├── volume_measures.txt         # Output volume measurements
└── Data/
    ├── Final_Volumes/          # Final SU volume meshes (OBJ)
    └── Pgram_Job_*/            # Per-job intermediate files and debug renders
```

## Prerequisites

### CloudCompare with Python Support (macOS Apple Silicon)

1. Download the CloudComPy binary from [openfields.fr](https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/8-archived-cloudcompy-binaries) — use the arm64 build.
2. Extract to `~/Desktop/CloudComPy310_clean/` using `ditto --norsrc` (not a regular unzip) to strip macOS quarantine attributes:
   ```bash
   ditto --norsrc ~/Downloads/CloudComPy310_<date>.zip ~/Desktop/CloudComPy310_clean/
   ```
3. Re-sign the Python binary to allow loading CloudComPy's libraries (required by macOS hardened runtime):
   ```bash
   cat > /tmp/cloudcompy_entitlements.plist << 'EOF'
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
       <true/>
       <key>com.apple.security.cs.disable-library-validation</key>
       <true/>
   </dict>
   </plist>
   EOF

   codesign --force --sign - --options runtime \
     --entitlements /tmp/cloudcompy_entitlements.plist \
     ~/miniconda3/envs/CloudComPy310/bin/python3.10
   ```

### Conda Environment

```bash
conda create --name CloudComPy310 python=3.10
conda activate CloudComPy310
conda config --add channels conda-forge
conda config --set channel_priority flexible
conda install -y boost cgal cmake draco "ffmpeg=6.1" gdal jupyterlab laszip \
  matplotlib "mysql=8" notebook numpy opencv "openssl=3.1" pcl pdal psutil \
  pybind11 quaternion "qhull=2020.2" "qt=5.15.8" scipy sphinx_rtd_theme \
  spyder tbb tbb-devel "xerces-c=3.2" xorg-libx11
```

### Running Scripts

Use `run.sh` (sets `PYTHONPATH` to include CloudComPy frameworks):

```bash
./run.sh pre_snip_script.py
./run.sh auto_snip_script.py
./run.sh post_snip_script.py
```

### Opening Files in CloudCompare GUI

Use `open -a` (LaunchServices). Direct binary invocation produces "no access right" errors on macOS. File > Open inside CloudCompare does not open a file picker.

```bash
# Files with spaces/commas in names must be copied to /tmp first
cp "Data/Pgram_Job_720_SU17007, 17008,17011_cleaned_su_17015.bin" /tmp/bottom.bin
open -a "~/Desktop/CloudComPy310_clean/CloudCompare/CloudCompare.app" /tmp/top.bin /tmp/bottom.bin
```

---

## Usage Workflow

### Step 1: Configure Input Models

Place PLY files in `~/Documents/TARP/ply/` with naming convention:
`Pgram_Job_<job_number>_SU<su_numbers>.ply`

Edit the JSON config for the season you're processing (e.g. `example-17000.json`):

```json
[
  {
    "top": "710",
    "bottom": "720",
    "annotations": ["SU_17015_annotation.png"]
  }
]
```

- `"top"` / `"bottom"`: job numbers matching PLY filenames
- `"annotations"`: list of annotation PNG filenames (one per SU in this pair)

### Step 2: Run Pre-snip

```bash
./run.sh pre_snip_script.py
```

Loads paired PLY meshes, samples them to point clouds, computes bidirectional C2C distances, and saves `*.clone_top_with_dist_*.bin` and `*.clone_bottom_with_dist_*.bin` in `Data/<top_job_folder>/`.

### Step 3: Run Auto-snip

```bash
./run.sh auto_snip_script.py
```

For each job pair, this script:

1. **Loads the pre-snip clouds** from Step 2.
2. **Renders a top-down orthographic image** of the top cloud (1 cm/px, RGB colors from photogrammetry). This render looks like the scene viewed from directly above.
3. **Extracts the yellow polygon** from the annotation PNG. The annotation image is a top-down aerial photo of the trench with three types of annotation lines drawn in pure, vibrant colors:
   - **Red + Blue**: together outline the full trench boundary
   - **Yellow**: outlines the specific SU being measured
   
   Yellow pixels are detected in HSV color space (S > 150 to exclude muted terrain colors). Connected-component analysis isolates the dominant cluster (the annotation line), and its convex hull becomes the SU polygon.

4. **Aligns annotation → render using PCA** (Principal Component Analysis):

   Feature matching (ORB/SIFT) fails here because the annotation photo and the orthographic render have a large rotation difference (~35°) and different visual characteristics. Instead, geometric alignment is used:

   - **PCA on red+blue outline pixels** (annotation): the annotation lines form an elongated blob matching the trench shape. PCA finds the blob's centroid, principal axis (long axis of the trench), and spread along each axis (trench length and width in pixel units).
   - **PCA on non-black pixels** (render): the point cloud footprint is the exact trench area. PCA finds its centroid, principal axis, and spread in render pixel units.
   - **Similarity transform**: since both PCAs describe the same physical trench, their parameters can be directly aligned:
     - **Rotation**: difference in principal axis angles (e.g. 73° annotation → -149° render = -222° rotation)
     - **Scale**: ratio of spreads (render length / annotation length ≈ 0.95)
     - **Translation**: after rotating and scaling around the annotation centroid, shift to land on the render centroid
   - The 180° ambiguity in PCA eigenvectors is resolved by trying both orientations and keeping whichever places the yellow polygon within the render image bounds.

5. **Transforms the yellow polygon** through the similarity transform → render pixel space → world XY coordinates (using the render's known pixel-to-world linear mapping).

6. **Crops both clouds** (top and bottom) to only keep points whose XY falls inside the world-space polygon, using a CloudComPy scalar field mask + `filterBySFValue`.

7. **Saves** `*_cleaned_su_<N>.bin` files in `Data/<top_job_folder>/` alongside debug images:
   - `debug_topdown_render.png`: the top-down render of the full cloud
   - `debug_SU<N>_render_overlay.png`: render with the transformed polygon drawn in green

### Step 4: Generate Final Volumes

```bash
./run.sh post_snip_script.py
```

For each matched `*_cleaned_su_<N>.bin` pair:
- Computes normals (inverts bottom cloud normals so they point inward)
- Merges top and bottom clouds
- Runs Poisson surface reconstruction (depth=11)
- Computes 3D mesh volume (`cc.ccMesh.computeMeshVolume`) in cm³
- Computes 2.5D projected volume (`cc.ComputeVolume25D`)
- Appends to `volume_measures.txt`
- Saves `Data/Final_Volumes/SU_<N>_raw.obj` and `Data/<top_folder>/SU_<N>_top_raw.obj`

---

## Output Files

| File | Description |
|------|-------------|
| `Data/<top_folder>/*_cleaned_su_<N>.bin` | Cropped point clouds for each SU |
| `Data/Final_Volumes/SU_<N>_raw.obj` | Merged Poisson mesh for volume calculation |
| `Data/<top_folder>/SU_<N>_top_raw.obj` | Top surface mesh |
| `Data/<top_folder>/<N>_post_snip.bin` | CloudCompare project file for review |
| `volume_measures.txt` | Tab-separated: SU name, 3D volume (cm³), 2.5D volume (m³), warnings |
| `Data/<top_folder>/debug_topdown_render.png` | Top-down render used for alignment |
| `Data/<top_folder>/debug_SU<N>_render_overlay.png` | Polygon overlay on render for QC |

---

## Annotation Image Format

Each `SU_<N>_annotation.png` is a top-down aerial photograph of the excavation trench with annotation lines:

- **Red polyline**: one side of the trench boundary
- **Blue polyline**: the other side (together red+blue = full trench)
- **Yellow polyline**: boundary of the specific SU being measured

The yellow line is drawn by the archaeologist to mark the horizontal extent of an SU on the surface. One annotation image per SU; multiple SUs in the same job pair each get their own annotation file listed in the JSON.

---

## Troubleshooting

**CloudComPy import fails ("library load disallowed by system policy")**: Re-run the `codesign` command from the Prerequisites section — the entitlements plist at `/tmp/` is ephemeral and may need to be recreated after a reboot.

**Yellow polygon not detected**: Check that the annotation PNG uses pure yellow (H=20-40°, S>150 in HSV). If the annotation tool uses a different color, update `color_ranges` in `extract_polygon_for_color`.

**PCA alignment clearly wrong**: Visually inspect `debug_SU<N>_render_overlay.png`. If the polygon is in the wrong location, the red+blue outlines in the annotation may not sufficiently cover the trench shape for PCA alignment. Consider using the C2C distance scalar field from the pre-snip output to identify high-difference regions as a cross-check.

**"no access right" when opening files in CloudCompare**: Use `open -a` (LaunchServices) rather than calling the binary directly. Copy files with special characters (spaces, commas) to `/tmp/` first.

**Memory issues during Poisson reconstruction**: Reduce `depth` parameter in `post_snip_script.py` from 11 to 9 or 10.

---

## Future Enhancements

- **C2C-based snipping**: Use the C2C distance scalar field already present in pre-snip bins to locate SUs geometrically — regions of anomalously high distance between surfaces correspond to SU locations, eliminating dependence on annotation image registration.
- **Multiple SUs per pair**: Already supported in JSON (`"annotations": ["SU_A.png", "SU_B.png"]`); each annotation produces its own cleaned pair.
- **Incremental processing**: Skip SUs already present in `Data/Final_Volumes/`.
