# TARP CloudCompare Script — Context for Agents

> **New machine?** See [pre-req.md](pre-req.md) for environment setup before running anything.
> Current active branch: `claude-vision-registration`. Changes uncommitted — see IMMEDIATE NEXT STEPS section.

---

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

### 10. PCA-Chamfer: PCA-locked rotation + Chamfer E-shape translation (**current best, in production**)
`register_lidar_to_ply_world_pca_chamfer()`

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

### 13. AKAZE-first cascade (multi-site validated, current production)

**Key finding from multi-site testing:** Sites differ in whether the LiDAR scan has texture correspondence with the PLY photogrammetry.
- **Sites with texture match** (e.g. SU20002/20003): AKAZE gets 33–36 RANSAC inliers. PreRotAKAZE is the correct primary method — it matches to the right place even in large multi-room scenes.
- **Sites without texture match** (e.g. SU22000, SU20005, SU21001): LiDAR sees wall faces from inside; PLY sees tops from above. AKAZE gets 2–4 inliers (noise). PCA-Chamfer is the fallback.

**Current cascade in `auto_snip_script.py`:**
1. Run `register_lidar_to_ply_world_prerot_akaze` → check `reg_debug["inliers"]`
2. If inliers ≥ 20 → use AKAZE result
3. If inliers < 20 AND `ANTHROPIC_API_KEY` set → fall back to `register_lidar_to_ply_world_claude_vision` (model=`claude-sonnet-4-6`)
4. If Claude Vision fails → fall back to `register_lidar_to_ply_world_pca_chamfer`
5. If chamfer fails → fall back to `register_lidar_to_ply_world` (RGB PCA)

**Why chamfer fails on large multi-room sites:** The PCA centroid of a large excavated area is in the middle of the whole scene, not near the specific room being annotated. Chamfer finds a low-cost local minimum matching the polygon to the wrong set of wall edges. AKAZE has no such problem because it matches features directly without needing a centroid prior.

---

### 14. Poisson density trimming (bubble fix, in production)

After Poisson reconstruction (`density=True`), phantom boundary faces at the crop edge have low vertex support → low density scalar. `cc.filterBySFValue` accepts a `ccMesh` and filters by the associated vertex cloud's active SF.

Trimming at p10 of the density distribution reduced SU22000_SU1 volume from ~19M cm³ (bubble-inflated) to ~607K cm³ (plausible).

---

### 15. Claude Vision — no-chamfer, best-of-3, quality gate (current production)

**Branch:** `claude-vision-registration` (branched from `automate-snipping-fix-shift`)

**Motivation:** PCA-Chamfer fails on hard sites (SU20005, SU21001) because the PCA centroid of the whole excavated area doesn't land near the specific sub-unit being annotated. Chamfer then finds false local minima on similar wall edges. Claude can semantically identify "which room in the aerial photo corresponds to what the iPhone scanned from inside" without texture correspondence.

**Function:** `register_lidar_to_ply_world_claude_vision()` in `auto_snip_lidar.py`

**Algorithm (current production):**
1. Build a side-by-side composite: left=LiDAR render (annotation polygon outlined in cyan, no fill), right=PLY render (CLAHE contrast-enhanced, clipLimit=3.0, tileGridSize=(8,8)). Both panels capped at 1024px long edge.
2. Compute `ann_frac` = fraction of LiDAR scan covered by annotation union mask. Pass to Claude as context.
3. Call `claude-haiku-4-5-20251001`. Model returns JSON: `{"cx", "cy", "w", "h", "reasoning"}` — all in right-panel-relative pixel coords (0..psW-1, 0..psH-1). cx/cy are clamped after parsing.
4. Scale cx_ply, cy_ply back to native PLY resolution using p_scale.
5. Lock PCA rotation. Check 180° flip.
6. **Skip Chamfer entirely** (`use_chamfer=False` always). Use Claude's bbox center directly as translation.
7. **Best-of-3 retries**: make 2 additional API calls, compute PLY-edge meanDist for each, keep the result with lowest meanDist. Haiku is stochastic; retrying gives multiple chances to get the correct semantic placement.
8. Return transform + `reg_debug["meanDist"]`.

**Quality gate in cascade (auto_snip_script.py ~line 430):**
- After Claude Vision returns, check `reg_debug["meanDist"]`.
- If `meanDist > 100px` → reject and fall through to PCA-Chamfer. High meanDist means polygons are far from any PLY wall edge → clearly wrong placement.
- If `meanDist ≤ 100px` → accept Claude Vision result.

**Why no Chamfer:**
Chamfer causes false local minima on BOTH single-polygon AND multi-polygon sites because the PLY has many similar wall edges everywhere. A low Chamfer cost does NOT imply correct registration — it only means the polygon boundary landed near SOME wall edge, not necessarily the correct one. Haiku's semantic placement (even at 20px meanDist) is more accurate than Chamfer's geometric refinement landing at the wrong wall.

**Why haiku instead of sonnet:**
Sonnet-4-6 truncated its JSON response at 512 tokens due to long reasoning, causing silent parse failures and cascade fallback to PCA-Chamfer. Haiku is faster, cheaper, and reliably produces valid JSON. Haiku's spatial accuracy is sufficient when Chamfer is not used.

**Per-site status (validated):**

| Site | Cascade method | MeanDist | Status |
|------|----------------|----------|--------|
| example-20002 (SU20002) | AKAZE, 36 inliers | — | ✅ CORRECT |
| example-20003 (SU20003) | AKAZE, 33 inliers | — | ✅ CORRECT |
| example-20005 (SU20005) | Claude Vision haiku no-chamfer | ~20–22px | ✅ CONFIRMED CORRECT by user |
| example-21001 (SU12001) | Claude Vision haiku no-chamfer | ~10.75px (when lucky) or falls to PCA-Chamfer 2.67px | ⚠️ STOCHASTIC — see notes |

**SU21001 behaviour:**
- Haiku's modal answer for 21001 places polygons in the upper PLY corridor (meanDist ≈ 294px → rejected by quality gate).
- On lucky runs (~1 in 3 first attempts), haiku sees "corridor-like structure, middle two-thirds" and gives cx=583, cy=706, meanDist=10.75px — this is CORRECT (all 3 rooms in the corridor at the right position).
- Best-of-3 means each run gets 3 chances. Probability of at least one good attempt ≈ 70%.
- If all 3 attempts fail quality gate → falls back to PCA-Chamfer (2.67px, slightly NW but visually reasonable).
- Visual check: `debug_SU12001_lidar_vs_result.png` should show 3 polygons spanning the full corridor structure. If only one polygon visible or polygons are in the upper portion → it's the fallback PCA-Chamfer result.

**Code locations:**
- `_call_claude_for_region()` → auto_snip_lidar.py ~line 1257: API call, max_tokens=512, response parse, coordinate clamping
- `register_lidar_to_ply_world_claude_vision()` → auto_snip_lidar.py ~line 1346. `use_chamfer=False` path has best-of-3 retry loop.
- Cascade integration → auto_snip_script.py ~line 430: always `use_chamfer=False`, `model="claude-haiku-4-5-20251001"`, quality gate on meanDist > 100px.
- Experimental block: runs claude_vision (haiku+Chamfer) and claude_vision_nochamfer (haiku, no-chamfer) for comparison.

**Environment:**
- `ANTHROPIC_API_KEY` must be set. Put in `.env` file (gitignored) at repo root.
- `run.sh` auto-sources `.env` if present.
- `.env.example` shows the format.

**Estimated credit cost:** ~$0.001–0.003 per SU for haiku (3 API calls: cascade + 2 retries; ~512 output tokens each).

**Debug images:**
- `debug_<SU>_claude_prompt.png` — exact composite image sent to Claude (for debugging what the model saw)
- `debug_<SU>_claude_vision_lidar_vs_result.png` — experimental claude_vision (Chamfer) result
- `debug_<SU>_claude_vision_nochamfer_lidar_vs_result.png` — experimental no-chamfer result

---

### 16. Multi-polygon support (all clusters shown in debug, crop uses union)

When a USDZ contains multiple yellow annotation clusters (e.g., SU21001 has 3 rooms), all clusters are detected and stored in `lidar["xz_polygons"]` (sorted largest first). The largest polygon is used as the registration anchor (`xz_polygon`). All polygons are:
- Drawn in `lidar_yellow.png` debug (green=anchor, cyan=secondary)
- Drawn in `lidar_vs_result.png` debug
- Passed to `register_lidar_to_ply_world_claude_vision()` via `xz_polygons=` kwarg
- Used in Chamfer cost (all polygon outlines, as of latest commit)
- Used in crop: `transform_fn` applied to each polygon, crop = union of all transformed polygons

---

## Key invariants

- `scale = 1.0` always. Both LiDAR and PLY are physical metres, same scene.
- Rotation from RGB PCA is correct for this site (-241.2° for SU22000). Lock it; don't re-search it.
- Threshold is a **fixed fraction of the elevation range** (`pct/100.0`), NOT a data percentile.
- `center_frac` parameters exist in the API but default to 1.0 (disabled). Only enable if you know the floor is geometrically central in both scans for your site.
- PCA 180° ambiguity: always check both 0° and 180° rotations and pick the one where the yellow polygon falls inside the PLY render bounds.
- Output dir is `Data/<json_id>/` (e.g. `Data/example-20002/`), NOT the PLY job folder. This prevents contamination when two JSONs share the same top job.
- Multiple yellow polygons in a USDZ are all detected; crop uses the union; registration uses the largest.
- Texture selection picks the **largest PNG by file size** from the USDZ (highest resolution).
- Claude Vision cx/cy must be RIGHT-PANEL-RELATIVE (0..psW-1). They are clamped after parsing. If you see cx values > psW in logs, Claude gave composite-image coords — the clamp corrects it but the seed is degraded.
- DO NOT fill the annotation polygon in the Claude prompt image. The natural yellow texture in the LiDAR render is what Claude needs; an opaque fill hides it and degrades matching.
- **Chamfer is DISABLED for Claude Vision** (`use_chamfer=False` always). Chamfer finds false local minima on both single- and multi-polygon sites — low meanDist after Chamfer ≠ correct registration. Use Claude's semantic center directly.
- **Quality gate**: if Claude Vision no-chamfer meanDist > 100px, the result is rejected (polygons far outside the PLY boundary) and cascade falls through to PCA-Chamfer.
- **Best-of-3 retries**: `_call_claude_for_region()` is called 3 times in the no-chamfer path; result with lowest meanDist is kept. Haiku is stochastic — 3 tries gives ~70% chance of a good result for hard sites.
- Claude Vision uses `claude-haiku-4-5-20251001` in the cascade. Sonnet-4-6 was dropped because it truncates its JSON at 512 tokens when reasoning is long, causing silent parse failures.
- The PLY covers the ENTIRE archaeological site; the LiDAR scan covers just one sub-unit. Do NOT expect the annotation to fill the same fraction of the PLY as it fills of the LiDAR.

## IMMEDIATE NEXT STEPS (for continuing agent)

Branch: `claude-vision-registration`. All four sites have been validated and changes are **uncommitted**.

### 1. Commit current state

All four sites are working:
- 20002, 20003: AKAZE ✓
- 20005: Claude Vision haiku no-chamfer ✓ (user confirmed)
- 21001: Claude Vision haiku no-chamfer ✓ (10.75px when lucky) or PCA-Chamfer fallback (2.67px, slightly NW)

```bash
git add auto_snip_lidar.py auto_snip_script.py CLAUDE.md
git commit -m "Claude Vision: drop Chamfer, haiku best-of-3 + quality gate"
```

### 2. Improve SU21001 reliability (optional)

21001 relies on haiku getting the "corridor" interpretation on one of 3 attempts (~70% success rate). When it fails, it falls back to PCA-Chamfer (2.67px, "slightly northwest").

Potential improvements:
- **Increase retries**: Change `range(2)` to `range(4)` in the retry loop for n_polys > 1 (5 total attempts → ~87% success rate)
- **Prompt hint for corridors**: If the PLY is taller than it is wide (corridor aspect ratio), add "This site is a long narrow corridor — the annotated rooms may span much of its length."
- **Manual override**: Add `registration_override: {"cx": 583, "cy": 706}` to the JSON input to hardcode the correct PLY coords for this site.

### 3. Run post-snip / volume pipeline

Once registration is confirmed, run `post_snip_script.py` on the newly cropped clouds to get volumes. Check for Poisson bubble artifacts (should be handled by density trimming at p10).

---

## Potential future work

- **Chamfer with fine rotation refinement:** A tight ±5° rotation refinement around the PCA value might squeeze out the last few pixels of error.
- **DEM ridge with tighter GeoTIFF threshold:** Top 2% of ridge cells (not 10%) would give a sparser, more distinctive PLY ridge pattern that might help.
- **Multi-scan AKAZE threshold tuning:** The 20-inlier threshold has been validated on 5 sites. May need adjustment for sites with unusual scan geometry.
- **Per-polygon volumes:** When a USDZ contains multiple yellow regions, currently they're combined into one crop. Could process each separately for per-sub-unit volumes.
- **Claude direct correspondences (no Chamfer):** Ask Claude to identify specific matched points (e.g., corners, stone block center) in both panels. Use those to compute the affine transform directly. Avoids false Chamfer local minima entirely.

## Debug images written per annotation

| File | Contents |
|------|----------|
| `debug_<SU>_lidar_yellow.png` | LiDAR render with yellow annotation highlighted; cyan=primary, green=secondary polygons |
| `debug_<SU>_lidar_vs_result.png` | Side-by-side: LiDAR (all polys in cyan/green) \| PLY with result polygons (green=primary, yellow=secondary) |
| `debug_<SU>_pca_axes.png` | PCA axis overlays on both renders for rotation debugging |
| `debug_<SU>_registration.png` | PLY top-down render with transformed annotation polygon in magenta |
| `debug_<SU>_snip_reference.png` | PLY render (left) + darkened render with ALL crop regions in green (right) |
| `debug_<SU>_claude_prompt.png` | Exact composite image sent to Claude (LiDAR left, CLAHE PLY right, grey divider) |
| `debug_<SU>_{method}_lidar_vs_result.png` | Same comparison for each experimental method |
| `debug_<SU>_claude_vision_lidar_vs_result.png` | Experimental claude_vision result (haiku, for comparison with cascade sonnet) |

## Branch history

| Branch | Content |
|--------|---------|
| `main` | Original working scripts |
| `lidar-annotation-strategy` | USDZ parsing, yellow face extraction, contour polygon |
| `lidar-dem-registration` | DEM-based PCA registration experiments |
| `automate-snipping-fix-shift` | All translation-fix experiments; chamfer matching; AKAZE cascade |
| `claude-vision-registration` | **CURRENT** — Claude Vision + Chamfer; multi-polygon support; sonnet-4-6 cascade |
