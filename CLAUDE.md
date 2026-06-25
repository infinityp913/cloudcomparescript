# TARP CloudCompare Script — Context for Agents

## What this repo does

Automates the "snipping" step in the TARP archaeology volume pipeline:

1. **pre_snip_script.py** — loads two PLY photogrammetry clouds (top + bottom of a stratigraphic unit), renders a top-down RGB image, used for context.
2. **auto_snip_script.py** — given a USDZ iPhone LiDAR scan containing a yellow-painted annotation region, registers it to the PLY world coordinate system and crops both clouds to that annotation shape.
3. **post_snip_script.py** — meshes the cropped clouds and computes volumes.

The hard part is step 2: **registering the LiDAR scan to the PLY world frame**.

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

../lidars/
  *.usdz                      # iPhone LiDAR scans (ignored by git, large)
```

---

## Coordinate systems

| Dataset | Frame | Notes |
|---------|-------|-------|
| iPhone USDZ | Local scanner, Y-up, metres | No GPS. `metersPerUnit=1`, `upAxis="Y"`. Origin is arbitrary scanner placement. |
| PLY cloud | Local metres from PLY bounding box origin | Same physical scene as GeoTIFF but stored with offset `(ply_x0, ply_y0)` subtracted |
| GeoTIFF DEM | UTM Zone 32N, absolute metres | `GetGeoTransform()` gives `(x0_utm, px_width, 0, y0_utm_top, 0, px_height_neg)`. Y origin is at TOP row (max northing), decreasing downward. |
| PLY world (our term) | Same as PLY cloud | Floor cells from GeoTIFF are converted: `local_x = utm_x - (dem_utm_x0 - ply_x0)`, `local_y = utm_y - (dem_utm_y_bot - ply_y0)` |

The iPhone USDZ has **no GPS or world transform** — registration is always required.

---

## Registration: all approaches tried

### 1. RGB footprint PCA (baseline, always available)
**File:** `auto_snip_lidar.py` → `register_lidar_to_ply_world()`, `_pca_footprint()`

PCA on non-black pixels in top-down renders of both the LiDAR and PLY scenes. Computes centre, main axis angle, and scale from the eigenvalues. 180° ambiguity resolved by checking the yellow polygon lands inside the PLY render bounds.

**Result:** Almost right — small but consistent offset. Scale can drift (0.985 for this site).

---

### 2. DEM from PLY point cloud (abandoned)
Tried building a top-down height grid from the PLY 3D cloud directly (max elevation per horizontal cell). Bad quality — PLY is noisy and the derived DEM doesn't match the GeoTIFF DEM geometry.

---

### 3. DEM-based PCA — wall tops, global threshold, scale from PCA spread (scale ≈ 0.85)
**File:** `auto_snip_lidar.py` → `register_lidar_to_ply_world_dem()`

Used provided GeoTIFF DEMs in `Data/DEMs/`. Selected the top 30% of elevation (wall/surface tops). Scale computed from PCA spread ratio.

**Why it failed:**
- Scale = 0.85: PCA spread ratio is unreliable when the two "wall top" footprints differ in shape (LiDAR captures interior walls; GeoTIFF captures unexcavated baulk + surrounding terrain).
- GeoTIFF top cells dominated by a large unexcavated baulk (big blob), unrelated to LiDAR top cells (scattered interior wall pattern). PCA centroid pulled to wrong location.

---

### 4. Fixed normalized threshold + scale = 1.0 (wall tops)
Same as above but:
- Threshold changed from data percentile (`np.nanpercentile(dem, 60)`) to fixed fraction of normalized range (`thresh_norm = pct / 100.0` applied to `dem_norm`). Key insight: `nanpercentile(dem, 60)` and `nanpercentile(dem_norm, 60)` select **identical cells** — normalisation is monotone. Must use a fixed fraction of the range, not a data percentile.
- Scale forced to 1.0 — both datasets are in physical metres covering the same scene.

**Result:** Still bad. Root cause unchanged: wall tops in GeoTIFF = baulk blob, wall tops in LiDAR = scattered interior geometry. No shape correspondence.

---

### 5. Floor approach — bottom 30% of normalized range, scale = 1.0 (**current best**)
`_compute_lidar_dem_wall_pts(..., use_floor=True, pct=30)`
`_load_geotiff_dem_wall_pts(..., use_floor=True, wall_pct=30)`

Selected the **lowest 30% of normalized elevation** in both DEMs instead of the tops.

**Why it works better:** Both datasets' floor-elevation cells correspond to the same physical surface — the excavated trench floor depression. They're geometrically compact and in consistent relative positions within their scans, giving a better PCA axis.

**Observed stats for SU22000_SU1:**
- LiDAR: 1,846 floor cells, centre XZ=(0.613, -6.562), angle=173.2°
- GeoTIFF: 178,503 floor cells (≤ 2.380 m abs), centre=(225.011, 817.276), angle=81.6°
- Registration: rotation=88.3°, scale=1.0
- Yellow polygon world extent: X=[213.343, 224.534] Y=[812.717, 820.323]
- Result: polygon within PLY render bounds, correct "E" shape with wall notches

**Known fragility:** LiDAR floor cells are sparse (1,846) and concentrated in one corner of the scan. The 88.3° rotation is approximately 90° — the LiDAR Z-axis maps to the PLY X-axis, which is geographically plausible but could fail if a future scanner is oriented differently.

---

### 6. Center crop — global threshold (failed)
Added `lidar_center_frac` and `ply_center_frac` params. Restricted floor selection to the central rectangle of each DEM before applying the global threshold.

**Finding:** Floor is NOT near the centre of either dataset for this site:
- LiDAR floor cells: upper-left corner of scan (row centroid ≈ row 49 out of 540 — near the scan boundary)
- GeoTIFF floor cells: right edge of surveyed area (col centroid ≈ 89% of width, col range 1745–2729 out of 2730)

Center 50% crop → 0 GeoTIFF floor cells. Center 80% LiDAR crop (r0=54) → 156 floor cells but borderline.

**Why:** The iPhone scans from *inside* the trench. The floor appears wherever the scanner was closest to the ground — not necessarily at the horizontal centre of the scan. The GeoTIFF covers the entire site; the excavated pit happens to be near one edge of the survey area, not centred.

---

### 7. Center crop — centre-local threshold (failed)
Same center crop, but threshold normalised relative to the **centre region's own min/max** rather than the global range. Idea: within the center, select the relatively lowest cells even if they're not globally low.

**Finding:** GeoTIFF center 50% minimum elevation = 2.422 m (above the global floor threshold of 2.380 m). The centre region contains no actual trench floor — it's entirely mid-elevation wall-base and baulk material. Centre-normalised selection picks cells at 2.4–3.1 m which have no geometric correspondence to the LiDAR centre cells.

Registration result: polygon mostly outside render bounds (upper-right), clearly worse than baseline.

---

## Key invariants

- `scale = 1.0` always. Both LiDAR and PLY are physical metres, same scene.
- Threshold is a **fixed fraction of the elevation range** (`pct/100.0`), NOT a data percentile.
- `center_frac` parameters exist in the API but default to 1.0 (disabled). Only enable if you know the floor is geometrically central in both scans for your site.
- PCA 180° ambiguity: always check both 0° and 180° rotations and pick the one where the yellow polygon falls inside the PLY render bounds.

## Debug images written per annotation

| File | Contents |
|------|----------|
| `debug_<SU>_lidar_yellow.png` | LiDAR render with yellow annotation highlighted and green contour polygon |
| `debug_<SU>_lidar_dem_img.png` | LiDAR DEM (grayscale, max-Y per XZ cell) with selected floor cells in red |
| `debug_<SU>_ply_dem_img.png` | GeoTIFF DEM (grayscale, centre-normalised) with selected floor cells in red |
| `debug_<SU>_registration.png` | PLY top-down render with transformed annotation polygon in magenta |
| `debug_<SU>_snip_reference.png` | PLY render (left) + darkened render with crop region in green (right) |

## Branch history

| Branch | Content |
|--------|---------|
| `main` | Original working scripts |
| `lidar-annotation-strategy` | USDZ parsing, yellow face extraction, contour polygon |
| `lidar-dem-registration` | All DEM-based PCA registration experiments (current) |
