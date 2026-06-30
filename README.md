# CloudCompare Script

## 3D Volume Analysis for Archaeological Stratigraphic Units

Part of the **[Tharros Archaeological Research Project (TARP)](https://air.ht.lu.se/s/tharros/page/home)** — an automated pipeline for converting 3D PLY models into volumetric measurements of archaeological Stratigraphic Units (SUs). The workflow combines automated preprocessing, automated snipping via image registration and point cloud analysis, and post-processing to generate accurate 3D volumes.

## Overview

The pipeline processes paired top and bottom 3D models (PLY files) representing archaeological layers to compute the volume of material between them.

1. **Pre-snip** (`pre_snip_script.py`): Loads PLY meshes, samples them to point clouds, computes bidirectional cloud-to-cloud (C2C) distances, and saves one `.bin` pair per SU (prefixed `SU<su>_`) with distance scalar fields.
2. **Manual snip** (CloudCompare): The operator opens the SU's pre-snip bin pair in CloudCompare, crops top and bottom to the SU boundary, and saves **both** cropped clouds into a single project bin named `<su>.bin` (e.g. `20001.bin`) in `Data/SU<su>/`. Auto-snip (`auto_snip_script.py`) is available but disabled in the dashboard.
3. **Post-snip** (`post_snip_script.py`): Reads `input.json`, finds each SU's combined `<su>.bin`, identifies the top and bottom clouds inside it (by the `Pgram_Job_<n>` number in each cloud's name, matched to `input.json`), merges them, runs Poisson surface reconstruction, and computes 3D and 2.5D volumes.

## Project Structure

```
cloudcomparescript/
├── pre_snip_script.py          # Step 1: distance computation
├── auto_snip_script.py         # Step 2: snipping (autosnip or manual)
├── auto_snip_lidar.py          # Registration library (all LiDAR methods)
├── post_snip_script.py         # Step 3: mesh generation and volume calculation
├── run.sh                      # Wrapper to run scripts with correct CloudComPy env
├── input.json                  # Canonical input — edit this for your run
├── example-*.json              # Per-site example configs
├── eval_methods.py             # Registration method evaluation script
├── volume_measures.txt         # Output volume measurements
└── Data/
    ├── Final_Volumes/          # Final SU volume meshes (OBJ)
    ├── eval/                   # eval_methods.py debug composites
    └── Pgram_Job_*/            # Per-job intermediate files and debug renders
```

---

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
./run.sh pre_snip_script.py input.json
./run.sh auto_snip_script.py input.json
./run.sh post_snip_script.py input.json
```

All three scripts are also importable Python modules:

```python
import pre_snip_script, auto_snip_script, post_snip_script

pre_snip_script.run_presnip_pipeline("input.json")
auto_snip_script.run_snip_pipeline("input.json")
post_snip_script.run_postsnip_pipeline("input.json")
```

---

## Usage Workflow

### Step 1: Configure input.json

Place PLY files in `~/Documents/TARP/ply/` with naming convention:
`Pgram_Job_<job_number>_SU<su_numbers>.ply`

Edit `input.json` for the SU you're processing:

```json
[
  {
    "top": "786",
    "bottom": "787",
    "su": "20002"
  }
]
```

- `"top"` / `"bottom"`: Pgram job numbers matching PLY filenames.
- `"su"`: SU identifier (may be a range like `"22044-22048"`). Names the per-SU output folder `Data/SU<su>/` and the output meshes. Written automatically by the dashboard; add manually when running scripts directly.
- `"annotations"` (auto_snip only): list of annotation file paths — mode is inferred from extension:
  - `.usdz` → autosnip (iPhone LiDAR scan with yellow-painted annotation)
  - `.png` → manual snip (annotated PLY ortho with black stroke outline)

### Step 2: Run Pre-snip

```bash
./run.sh pre_snip_script.py input.json
```

Loads paired PLY meshes, samples them to point clouds, computes bidirectional C2C distances, and saves one pair per SU, `SU<su>_*_top_with_dist_*.bin` and `SU<su>_*_bottom_with_dist_*.bin`, in `Data/SU<su>/`.

### Step 3: Manually Snip in CloudCompare

After pre-snip, open the SU's bin pair in CloudCompare (use the **Open in CC** button in the dashboard) and crop both the top and bottom clouds to the SU boundary. Then save **both** cropped clouds together into a **single project bin** in `Data/SU<su>/`:

- `Data/SU<su>/<su>.bin` — e.g. `Data/SU20001/20001.bin`

The filename is flexible: the SU number with an optional `SU`/`su` prefix, in any case — `20001.bin`, `SU20001.bin`, and `su20001.bin` are all accepted. (Select both cropped clouds in the DB tree → *File ▸ Save* → name it `<su>.bin`.) Post-snip identifies which cloud is top vs bottom from the `Pgram_Job_<n>` number in each cloud's name (matched against `top`/`bottom` in `input.json`), so the `top`/`bottom` group labels don't need to be preserved. If you re-snip, just save again — the newest matching bin by modification time is used.

> **Auto-snip** (`auto_snip_script.py`) is still available for USDZ LiDAR or annotated ortho PNG workflows but is disabled in the dashboard. See `CLAUDE.md` for details.

### Step 4: Generate Final Volumes

```bash
./run.sh post_snip_script.py input.json
```

For each SU's combined `<su>.bin` in `Data/SU<su>/`:
- Loads the bin and identifies the top and bottom clouds by Pgram number
- Filters out low-C2C-distance fringe points (p25) that cause Poisson bubble artifacts
- Computes normals (inverts bottom cloud normals to point inward)
- Merges top and bottom clouds
- Runs Poisson surface reconstruction (depth=11) with density trimming at p10 (removes phantom boundary faces)
- Computes 3D mesh volume in cm³ and 2.5D projected volume in m³
- Appends to `volume_measures.txt`
- Saves `Data/Final_Volumes/SU_<N>_raw.obj` and `Data/SU<N>/SU_<N>_top_raw.obj`

---

## Output Files

| File | Description |
|------|-------------|
| `Data/SU<su>/SU<su>_*_top_with_dist_for_*.bin` | Top cloud — pre-snip output, opened by the operator for snipping |
| `Data/SU<su>/SU<su>_*_bottom_with_dist_for_*.bin` | Bottom cloud — pre-snip output, opened by the operator for snipping |
| `Data/SU<su>/<su>.bin` | Combined snip — both cropped clouds, saved by the operator (post-snip input) |
| `Data/SU<su>/<su>_post_snip.bin` | Post-snip project: merged + top/bottom clouds and meshes |
| `Data/Final_Volumes/SU_<su>_raw.obj` | Merged Poisson mesh for volume calculation |
| `Data/SU<su>/SU_<su>_top_raw.obj` | Top surface mesh |
| `volume_measures.txt` | Tab-separated: SU name, 3D volume (cm³), 2.5D volume (m³), warnings |
| `Data/<top_id>/debug_topdown_render.png` | Top-down PLY render used for alignment (auto_snip) |
| `Data/<top_id>/debug_SU<N>_{method}_lidar_vs_result.png` | LiDAR vs PLY comparison per method (auto_snip) |
| `Data/<top_id>/debug_SU<N>_snip_reference.png` | Crop region overlay on PLY render (auto_snip) |

---

## Troubleshooting

**CloudComPy import fails ("library load disallowed by system policy")**: Re-run the `codesign` command from the Prerequisites section — the entitlements plist at `/tmp/` is ephemeral and may need to be recreated after a reboot.

**Autosnip result is in the wrong location**: Inspect the `debug_SU<N>_{method}_lidar_vs_result.png` images for all 5 methods. If none are correct, switch to manual snip: draw the crop boundary in black on `debug_topdown_render.png` and re-run.

**No annotation polygon found in manual snip**: Ensure black strokes are truly black (all channels < 50). Increase the morphological close kernel in `run_manual_snip()` if stroke gaps are large.

**"no access right" when opening files in CloudCompare**: Use `open -a` (LaunchServices) rather than calling the binary directly. Copy files with special characters (spaces, commas) to `/tmp/` first.

**Memory issues during Poisson reconstruction**: Reduce `depth` parameter in `post_snip_script.py` from 11 to 9 or 10.

---

## API / Vision Methods (Disabled)

Claude Vision (`claude-haiku-4-5-20251001`) and OpenRouter model methods are preserved in `auto_snip_lidar.py` as `_DISABLED_call_claude_for_region` and `_DISABLED_register_lidar_to_ply_world_claude_vision`. To re-enable, remove the `_DISABLED_` prefix from the function names and set `ANTHROPIC_API_KEY` in `.env`.

Evaluation across 4 sites showed `gemini-2.5-flash` (via OpenRouter) achieves 1.39 m mean centroid error — essentially tied with `rgb_pca` (1.30 m) but at API cost. See `eval_methods.py` for the full evaluation framework.
