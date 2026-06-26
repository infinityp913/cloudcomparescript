# Prerequisites — New Machine Setup

Before running any scripts, ensure the following are in place.

---

## 1. CloudComPy environment

The scripts run inside a CloudComPy conda environment that bundles CloudCompare's Python bindings.

**Install location expected by `run.sh`:**
```
~/Desktop/CloudComPy310_clean/          # CloudComPy root
~/miniconda3/envs/CloudComPy310/        # conda env
```

If your paths differ, edit the top of `run.sh`:
```bash
CLOUDCOMPY_ROOT=~/Desktop/CloudComPy310_clean
CONDA_PYTHON=~/miniconda3/envs/CloudComPy310/bin/python3.10
```

CloudComPy for Apple Silicon (arm64) can be downloaded from the CloudComPy releases page. The env name must be `CloudComPy310` and use Python 3.10.

---

## 2. Python packages inside the conda env

The `anthropic` SDK is required for Claude Vision registration. Install it into the CloudComPy env (not the system Python):

```bash
~/miniconda3/envs/CloudComPy310/bin/pip install anthropic
```

All other dependencies (opencv-python, numpy, gdal, etc.) come bundled with CloudComPy.

---

## 3. Anthropic API key

Create a `.env` file in the repo root (gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

`run.sh` auto-sources this file. Without it, the Claude Vision registration step is skipped and the cascade falls back to PCA-Chamfer.

See `.env.example` for the format.

---

## 4. Data directories

The following directories are gitignored (large files). Copy them alongside the repo:

| Path | Contents | Required for |
|------|----------|--------------|
| `Data/Pgram_Job_*/` | CloudCompare binary point clouds (.bin) | auto_snip, post_snip |
| `Data/DEMs/` | GeoTIFF DEMs (.tif) | DEM-based registration methods |
| `../lidars/` | iPhone LiDAR scans (.usdz) — **one level above repo root** | auto_snip |

The `../lidars/` folder must sit at the same level as the repo root, e.g.:
```
TARP/
  cloudcomparescript/     ← repo root
  lidars/
    tarpf24726.usdz
    tarpf24441.usdz
    ...
```

---

## 5. Running scripts

All scripts must be launched via `run.sh` (sets up PYTHONPATH and loads .env):

```bash
cd cloudcomparescript
./run.sh pre_snip_script.py example-20002.json
./run.sh auto_snip_script.py example-20002.json
./run.sh post_snip_script.py example-20002.json
```

Do **not** run with system Python — CloudComPy bindings won't be available.

---

## 6. macOS security (arm64 only)

On first run, macOS may block CloudComPy dylibs. If you see `killed` or permission errors:

```bash
xattr -cr ~/Desktop/CloudComPy310_clean
```

Then re-run. You may also need to allow it in System Settings → Privacy & Security.
