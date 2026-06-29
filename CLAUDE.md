# TARP CloudCompare Script — Agent Reference

> **New machine?** See [pre-req.md](pre-req.md) for environment setup before running anything.
> **Current branch:** `lidar-dem-registration`

---

## What this repo does

Automates the "snipping" step in the TARP archaeology volume pipeline:

1. **pre_snip_script.py** — loads two PLY photogrammetry clouds (top + bottom of a stratigraphic unit), computes cloud-to-cloud distances, and saves the tagged clouds.
2. **auto_snip_script.py** — given either a USDZ iPhone LiDAR scan (autosnip) or a hand-annotated ortho PNG (manual snip), locates the annotation in the PLY world frame and crops both clouds to that region.
3. **post_snip_script.py** — merges the cropped clouds, runs Poisson reconstruction, and computes volumes.

---

## Callable Python API

All three scripts are importable modules. The canonical input is `input.json`.

```python
import pre_snip_script
import auto_snip_script
import post_snip_script

pre_snip_script.run_presnip_pipeline("input.json")
auto_snip_script.run_snip_pipeline("input.json")
post_snip_script.run_postsnip_pipeline("input.json")
```

Each script can also be run directly:
```bash
./run.sh pre_snip_script.py input.json
./run.sh auto_snip_script.py input.json
./run.sh post_snip_script.py input.json
```

---

## input.json format

```json
[
  {
    "top": "786",
    "bottom": "787",
    "annotations": ["../lidars/tarpf24477.usdz"]
  }
]
```

- `top` / `bottom`: Pgram job numbers used to locate the `.ply` files under `~/Documents/TARP/ply/`.
- `annotations`: list of annotation file paths. Mode is inferred from extension:
  - `.usdz` → **autosnip** (LiDAR scan with yellow-painted annotation)
  - `.png` → **manual snip** (annotated PLY ortho with black stroke outline)
- For manual snip, add `"su": "20002"` at the job level if the SU number can't be parsed from the filename.

Multiple annotations per job are supported. Multi-USDZ (multiple yellow clusters) is also supported — crop uses the union of all polygons.

---

## Two modes

### Mode 1 — Autosnip (`.usdz`)

Runs 5 math-based registration methods. `rgb_pca` is the default used for the actual crop; all 5 generate side-by-side debug composites for comparison.

| Method | Function in auto_snip_lidar.py | Notes |
|--------|-------------------------------|-------|
| `rgb_pca` | `register_lidar_to_ply_world` | **DEFAULT** — PCA on RGB footprints |
| `dist_pca` | `register_lidar_to_ply_world_dist_pca` | Distance-weighted PCA |
| `phase_corr` | `register_lidar_to_ply_world_phase_corr` | Phase-correlation on Canny edges |
| `prerot_akaze` | `register_lidar_to_ply_world_prerot_akaze` | Pre-rotated AKAZE feature matching |
| `pca_chamfer` | `register_lidar_to_ply_world_pca_chamfer` | PCA rotation + Chamfer translation |

Debug images written per method: `debug_SU{su}_{method}_lidar_vs_result.png`

**API methods** (Claude Vision, OpenRouter) are preserved in `auto_snip_lidar.py` as `_DISABLED_*` functions. Remove the `_DISABLED_` prefix to re-enable.

### Mode 2 — Manual Snip (`.png`)

Takes a hand-annotated PLY ortho image (black strokes drawn over the top-down render). Detects black pixels → morphological fill → maps ortho pixel coords to PLY world coords via content-bbox normalisation.

Debug image: `debug_SU{su}_manual_annotation.png` — ortho with extracted polygon in green.

---

## Registration method evaluation results

Evaluated across 4 sites (20002, 20003, 20005, 21001) using GT from annotated ortho diffs.

| Rank | Method | Mean error (m) | Notes |
|------|--------|---------------|-------|
| 1 | `rgb_pca` | 1.299 | Baseline — wins overall |
| 2 | `dist_pca` | 1.335 | Distance-weighted variant |
| 3 | `gemini_25flash` | 1.388 | OpenRouter vision LLM (disabled) |
| 4 | `claude_haiku_chamfer` | ~1.78 | Claude haiku + Chamfer (disabled) |
| 8 | `phase_corr` | 2.564 | Phase correlation |
| 9 | `prerot_akaze` | 2.630 | AKAZE — fails when LiDAR ≠ PLY texture |
| 10 | `pca_chamfer` | 2.861 | Chamfer without vision seed finds false minima |

**Key finding**: `rgb_pca` wins overall. PCA on RGB footprints works because LiDAR and PLY footprints share similar overall shape at these sites. `prerot_akaze` fails on sites where LiDAR sees interior wall faces and PLY sees tops from above (< 5 RANSAC inliers). Full eval results in `Data/eval/ranking.txt`.

---

## Data layout

```
Data/
  Pgram_Job_<id>_<SU>/        # photogrammetry job folders (ignored by git, large)
    *.bin                     # CloudCompare binary clouds
    debug_*.png               # debug images written by auto_snip_script.py
  DEMs/
    Pgram_Job_<id>_<SU>_dem.tif   # GeoTIFF DEM for each photogrammetry job
  Final_Volumes/              # output meshes (ignored by git)
  eval/                       # eval_methods.py output composites

../lidars/
  *.usdz                      # iPhone LiDAR scans (ignored by git, large)
```

---

## Coordinate systems

| Dataset | Frame | Notes |
|---------|-------|-------|
| iPhone USDZ | Local scanner, Y-up, metres | No GPS. Origin is arbitrary scanner placement. |
| PLY cloud | Local metres from PLY bounding box origin | Same scene as GeoTIFF but with offset subtracted |
| GeoTIFF DEM | UTM Zone 32N, absolute metres | Y origin at TOP row (max northing), decreasing downward |
| PLY world (our term) | Same as PLY cloud | GeoTIFF cells converted via `local_x = utm_x - (dem_utm_x0 - ply_x0)` |

The iPhone USDZ has **no GPS** — registration is always required for autosnip.

---

## Key invariants

- `scale = 1.0` always. Both LiDAR and PLY are physical metres, same scene.
- PCA 180° ambiguity: always check both rotations; pick the one where the yellow polygon falls inside PLY render bounds.
- Output dir is `Data/<json_id>/` (e.g. `Data/input/`), NOT the PLY job folder. This prevents contamination when two JSONs share the same top job.
- Multiple yellow polygons in a USDZ are all detected; crop uses the union; registration uses the largest.
- Texture selection picks the **largest PNG by file size** from the USDZ (highest resolution).
- Content-bbox normalisation for manual snip: non-black region of ortho ↔ PLY world bbox; Y is flipped (image top → world max Y).
- Claude Vision / OpenRouter methods are `_DISABLED_*` in `auto_snip_lidar.py`. Re-enable by removing the prefix. Requires `ANTHROPIC_API_KEY` or `OPEN_ROUTER_KEY` in `.env`.

---

## Debug images written per annotation

| File | Contents |
|------|----------|
| `debug_topdown_render.png` | Top-down PLY render used for registration |
| `debug_lidar_render.png` | LiDAR scan render (autosnip only) |
| `debug_SU{su}_lidar_yellow.png` | LiDAR render with yellow annotation highlighted |
| `debug_SU{su}_{method}_lidar_vs_result.png` | Side-by-side: LiDAR (cyan poly) \| PLY (green placed poly) — one per method |
| `debug_SU{su}_manual_annotation.png` | Ortho with extracted polygon overlaid (manual snip only) |
| `debug_SU{su}_snip_reference.png` | Full PLY render (left) + dimmed render with crop region in green (right) |

---

## Environment

- Python environment: `conda activate CloudComPy` (see [pre-req.md](pre-req.md))
- Run scripts via `run.sh` (sets up the correct conda env): `./run.sh auto_snip_script.py input.json`
- API keys: put in `.env` at repo root (gitignored). `run.sh` auto-sources it.
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  OPEN_ROUTER_KEY=sk-or-...
  ```
- `.env.example` shows the format.

---

## Evaluation script

`eval_methods.py` runs all registration methods against annotated GT and ranks them by centroid error.

```bash
./run.sh eval_methods.py              # full run (all methods)
./run.sh eval_methods.py --or-only   # OpenRouter models only
./run.sh eval_methods.py --new-only  # newly-added OR models only
```

Results: `Data/eval/ranking.txt`, per-site composites in `Data/eval/`.
