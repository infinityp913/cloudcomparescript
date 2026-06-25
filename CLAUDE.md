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

### 5. Floor approach — bottom 30% of normalized range, scale = 1.0
`_compute_lidar_dem_wall_pts(..., use_floor=True, pct=30)`
`_load_geotiff_dem_wall_pts(..., use_floor=True, wall_pct=30)`

Selected the **lowest 30% of normalized elevation** in both DEMs instead of the tops.

**Why it works better:** Both datasets' floor-elevation cells correspond to the same physical surface — the excavated trench floor depression. They're geometrically compact and in consistent relative positions within their scans, giving a better PCA axis.

**Observed stats for SU22000_SU1:**
- LiDAR: 1,846 floor cells, centre XZ=(0.613, -6.562), angle=173.2°
- GeoTIFF: 178,503 floor cells (≤ 2.380 m abs), centre=(225.011, 817.276), angle=81.6°
- Registration: rotation=88.3°, scale=1.0

**Known fragility:** LiDAR floor cells are sparse (1,846) and concentrated in one corner of the scan, not the interior. Floor centroid is at the scan boundary (render row 99 of 1512), making it an unstable translation anchor. The PCA rotation is approximately 90° which is plausible geographically but could fail if a future scanner is oriented differently.

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

### 8. Hybrid: PCA rotation + DEM floor centroid translation (failed)
`register_lidar_to_ply_world_dem_center()`

Used PCA to get rotation, then used DEM floor centroids (bottom 30% elevation) as the translation anchor instead of the PCA centroid.

**Why it failed:** LiDAR floor centroid is at render pixel (602, 99) — row 99 of 1512, right at the top scan boundary. The iPhone scanner is placed *inside* the trench so the floor appears at the scan edge, not the centre. This makes the floor centroid an unstable anchor that pushes the annotation off-screen (top-right).

---

### 9. Four appearance-based experimental methods (all failed — wrong tool)
**Branch:** `automate-snipping-fix-shift`

Tried in the comparison block: phase-correlation on Canny edges, pre-rotated AKAZE feature matching, annotation-boundary phase correlation, distance-weighted PCA centroid.

**Root cause of all failures:** The LiDAR scanned from *inside* the trench and sees wall **faces**; the photogrammetry sees wall **tops** from above. The two renders genuinely do not look alike:
- AKAZE got 4 RANSAC inliers (effectively nothing)
- Phase-correlation response ≈ 0.005 (noise floor)
- Appearance-based matching is the fundamentally wrong tool for this modality pair

**Distance-weighted PCA** (pixels weighted by squared distance from scan boundary) was the best of this batch — it got closer to the right position than standard PCA by downweighting peripheral scan extent that differs between modalities.

---

### 10. Chamfer E-shape matching + PCA-locked rotation (**current best, in production**)
`register_lidar_to_ply_world_chamfer()`

**Key insight:** Rotation is correct from PCA (-241.2° for SU22000_SU1); only translation is wrong. And the annotation E-polygon *is* the wall outline — its boundary should land on PLY wall edges.

Algorithm:
1. PCA on both renders → rotation `ang_pl - ang_li`. Correct 180° flip via polygon-within-bounds check.
2. **Lock rotation** to the PCA value (no rotation search).
3. Build PLY Canny edge image → distance transform `dt`.
4. Densify E-polygon into ~3px-spaced points; cost = mean `dt` at transformed points.
5. Coarse-to-fine 2-DOF translation search: ±300px coarse (20px steps), ±20px fine (4px steps).

**Result for SU22000_SU1:** meanDist=7.09px, annotation sits squarely in the excavation with the E-shape tracing wall edges. Top crop: 1.57M pts (vs 1.1M with baseline). Visually correct.

**Why it works:** Reducing from 3-DOF (rotation + translation) to 2-DOF (translation only) eliminates the clutter problem — even with dense PLY wall edges everywhere, a 2D slide of the E-shape finds the unique pose where ALL polygon edges align simultaneously.

---

### 11. DEM gradient-ridge matching (ran but less accurate than chamfer)
`register_lidar_to_ply_world_dem_ridge()`

Builds slope-magnitude images from LiDAR height grid and GeoTIFF DEM. Walls appear as high-gradient ridges in both. Chamfer-searches LiDAR ridge points against PLY ridge distance transform.

**Problem:** GeoTIFF produces 381,003 ridge cells (top 10% of all slopes) — far too many to give a selective signal. LiDAR only has 239 ridge cells. The imbalance makes the distance transform too dense and the cost surface flat. Result at translation-only search: meanDist=7.67px vs chamfer's 7.09px, with slight annotation clip at the render top.

**Potential improvement:** Threshold the GeoTIFF ridges more aggressively (top 2-3%) or restrict to the area of the site where the trench is.

---

### 12. Mutual information (failed — modality coupling too weak)
`register_lidar_to_ply_world_mutual_info()`

Maximised MI between downsampled PLY and warped LiDAR gray images. MI=0.101 — extremely low, confirming that faces-vs-tops have essentially no pixel-level statistical coupling. Result was worse than baseline (annotation pushed off top-right edge). Dead end for this modality pair.

---

## Key invariants

- `scale = 1.0` always. Both LiDAR and PLY are physical metres, same scene.
- Rotation from RGB PCA is correct for this site (-241.2°). Lock it; don't re-search it.
- Threshold is a **fixed fraction of the elevation range** (`pct/100.0`), NOT a data percentile.
- `center_frac` parameters exist in the API but default to 1.0 (disabled). Only enable if you know the floor is geometrically central in both scans for your site.
- PCA 180° ambiguity: always check both 0° and 180° rotations and pick the one where the yellow polygon falls inside the PLY render bounds.

## Potential future work

- **Chamfer with fine rotation refinement:** Now that translation is right, a tight ±5° rotation refinement around the PCA value might squeeze out the last few pixels of error.
- **DEM ridge with tighter GeoTIFF threshold:** Top 2% of ridge cells (not 10%) would give a sparser, more distinctive PLY ridge pattern that might help.
- **Multi-scan validation:** Test on a different USDZ (different scanner orientation). If PCA rotation fails on a new scan, chamfer's free-rotation search can be re-enabled with a tighter angular range (±30° around PCA) rather than ±20°.
- **Bubble removal in Poisson output:** Higher C2C percentile filter makes bubbles worse (removes too many boundary-constraining points). Better approach may be density-scalar trimming after Poisson, or filtering by normal Z component to remove near-vertical phantom faces.

## Debug images written per annotation

| File | Contents |
|------|----------|
| `debug_<SU>_lidar_yellow.png` | LiDAR render with yellow annotation highlighted and green contour polygon |
| `debug_<SU>_lidar_vs_result.png` | Side-by-side: LiDAR annotation (cyan) \| PLY with result polygon (green) |
| `debug_<SU>_pca_axes.png` | PCA axis overlays on both renders for rotation debugging |
| `debug_<SU>_registration.png` | PLY top-down render with transformed annotation polygon in magenta |
| `debug_<SU>_snip_reference.png` | PLY render (left) + darkened render with crop region in green (right) |
| `debug_<SU>_{method}_lidar_vs_result.png` | Same comparison for each experimental method |

## Branch history

| Branch | Content |
|--------|---------|
| `main` | Original working scripts |
| `lidar-annotation-strategy` | USDZ parsing, yellow face extraction, contour polygon |
| `lidar-dem-registration` | DEM-based PCA registration experiments |
| `automate-snipping-fix-shift` | All translation-fix experiments; chamfer matching (current production) |
