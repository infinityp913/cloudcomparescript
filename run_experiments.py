"""
Experimental registration methods for the auto-snip LiDAR pipeline.

Usage:  ./run.sh run_experiments.py example-22000.json

Runs two experiments for every USDZ annotation in the JSON file and writes
all debug images to sub-folders inside the job's output directory:

  Data/Pgram_Job_<top_id>/exp_circle/   — circle-centred DEM PCA
  Data/Pgram_Job_<top_id>/exp_icp/      — ICP refinement after RGB PCA

Neither experiment modifies the main pipeline outputs.
"""

import cloudComPy as cc
import cv2
import numpy as np
import json
import os
import sys
from scipy.ndimage import minimum_filter

from pre_snip_script import DATA_DIR, find_mesh_by_pgram_job, INPUT_MESH_PATH
import auto_snip_lidar

cc.initCC()

json_filepath = sys.argv[1] if len(sys.argv) > 1 else "example-22000.json"
with open(json_filepath) as f:
    job_data = json.load(f)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_bins(top_id, bottom_id):
    import glob
    search = os.path.join(DATA_DIR, top_id)
    top_bin = bottom_bin = None
    for b in glob.glob(os.path.join(search, "*.bin")):
        name = os.path.basename(b).lower()
        if "top_with_dist" in name:
            top_bin = b
        elif "bottom_with_dist" in name:
            bottom_bin = b
    if not top_bin or not bottom_bin:
        raise FileNotFoundError(f"Pre-snipped .bin pair not found in {search}")
    return top_bin, bottom_bin


def _render_topdown(cloud, resolution=0.01):
    coords = cloud.toNpArrayCopy()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0, y0, x1, y1 = float(x.min()), float(y.min()), float(x.max()), float(y.max())
    W = int(np.ceil((x1 - x0) / resolution)) + 1
    H = int(np.ceil((y1 - y0) / resolution)) + 1
    img = np.zeros((H, W, 3), dtype=np.uint8)
    px = np.clip(((x - x0) / resolution).astype(int), 0, W - 1)
    py = np.clip((H - 1 - (y - y0) / resolution).astype(int), 0, H - 1)
    order = np.argsort(z)
    if cloud.hasColors():
        rgba = cloud.colorsToNpArrayCopy()
        img[py[order], px[order]] = rgba[order, :3][:, ::-1]
    if img.max() == 0:
        z_n = ((z - z.min()) / max(z.max() - z.min(), 1e-6) * 255).astype(np.uint8)
        g = np.stack([z_n, z_n, z_n], axis=1)
        img[py[order], px[order]] = g[order]
    img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    return img, (x0, y0, x1, y1)


def _snip_reference(render_img, render_world_bbox, yellow_world):
    """Left panel = full render. Right = dimmed + green crop polygon."""
    rH, rW = render_img.shape[:2]
    rx0, ry0, rx1, ry1 = render_world_bbox
    px = np.clip(((yellow_world[:, 0] - rx0) / (rx1 - rx0) * rW).astype(int), 0, rW-1)
    py = np.clip(((ry1 - yellow_world[:, 1]) / (ry1 - ry0) * rH).astype(int), 0, rH-1)
    pts = np.stack([px, py], axis=1).reshape(-1, 1, 2)
    mask = np.zeros((rH, rW), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    hi = render_img.copy()
    hi[mask == 0] = (hi[mask == 0].astype(float) * 0.35).astype(np.uint8)
    cv2.polylines(hi, [pts], True, (0, 255, 0), 4)
    return np.hstack([render_img, hi])


def _save_imgs(folder, files: dict):
    """Save {filename: image_array} to folder, creating it if needed."""
    os.makedirs(folder, exist_ok=True)
    for name, img in files.items():
        if img is not None:
            path = os.path.join(folder, name)
            cv2.imwrite(path, img)
            print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Running experiments...")

for job in job_data:
    top_job    = job["top"]
    bottom_job = job["bottom"]
    annotations = job.get("annotations", [])

    top_id    = find_mesh_by_pgram_job(top_job,    INPUT_MESH_PATH)
    bottom_id = find_mesh_by_pgram_job(bottom_job, INPUT_MESH_PATH)
    if not top_id or not bottom_id:
        print(f"Cannot find PLY for top={top_job} / bottom={bottom_job}, skipping.")
        continue

    print(f"\nJob: top={top_id}  bottom={bottom_id}")

    top_bin, _ = _find_bins(top_id, bottom_id)
    top_cloud  = cc.loadPointCloud(top_bin)
    print(f"  Loaded top cloud: {top_cloud.size()} pts")

    print("  Rendering top-down image...")
    render_img, render_world_bbox = _render_topdown(top_cloud)

    dem_path = os.path.join(DATA_DIR, "DEMs", f"{top_id}_dem.tif")
    if not os.path.exists(dem_path):
        print(f"  No GeoTIFF DEM at {dem_path} — skipping DEM experiments")
        dem_path = None

    output_dir = os.path.join(DATA_DIR, top_id)

    for annotation_path in annotations:
        if not annotation_path.lower().endswith(".usdz"):
            continue

        print(f"\n  Annotation: {annotation_path}")
        try:
            lidar = auto_snip_lidar.process_usdz(annotation_path)
        except Exception as e:
            print(f"  Error processing USDZ: {e}")
            continue

        su = lidar["su_name"]
        print(f"  SU: {su}")

        # ------------------------------------------------------------------
        # Circle DEM PCA experiments at three percentiles
        # ------------------------------------------------------------------
        for circle_pct, circle_tag in [(70, "exp_circle_pct70"), (80, "exp_circle_pct80")]:
            if not dem_path:
                print(f"\n  [Circle pct={circle_pct}] Skipped (no GeoTIFF DEM)")
                continue
            print(f"\n  [Circle pct={circle_pct}] Circle DEM PCA (diameter=shorter/2)")
            c_folder = os.path.join(output_dir, circle_tag)
            try:
                tfm_c, reg_c, note_c, dem_debug_c = \
                    auto_snip_lidar.register_lidar_to_ply_world_circle(
                        lidar["lidar_pts"], lidar["lidar_xz_bbox"],
                        dem_path, render_world_bbox,
                        lidar["lidar_render"], render_img,
                        lidar["xz_polygon"],
                        pct=circle_pct,
                    )
                yellow_world_c = tfm_c(lidar["xz_polygon"])
                print(f"  Circle pct={circle_pct} polygon: "
                      f"X=[{yellow_world_c[:,0].min():.3f},{yellow_world_c[:,0].max():.3f}] "
                      f"Y=[{yellow_world_c[:,1].min():.3f},{yellow_world_c[:,1].max():.3f}]")
                _save_imgs(c_folder, {
                    f"SU{su}_lidar_dem.png":      dem_debug_c.get("lidar_dem_img"),
                    f"SU{su}_ply_dem.png":        dem_debug_c.get("ply_dem_img"),
                    f"SU{su}_registration.png":   reg_c,
                    f"SU{su}_snip_reference.png": _snip_reference(
                                                      render_img, render_world_bbox,
                                                      yellow_world_c),
                })
            except Exception as e:
                print(f"  Circle pct={circle_pct} experiment failed: {e}")

        # ------------------------------------------------------------------
        # Min-pool circle DEM PCA experiments
        # Apply spatial minimum_filter to DEM before threshold selection.
        # Pool size determines how far low-elevation regions spread.
        # pct=60 fixed; vary pool size: 5, 15, 30, 60 px.
        # ------------------------------------------------------------------
        if dem_path:
            from osgeo import gdal as _gdal
            _gdal.UseExceptions()

            # --- build LiDAR DEM once ---
            _cs = 0.02
            _x  = lidar["lidar_pts"][:, 0].astype(np.float32)
            _z  = lidar["lidar_pts"][:, 2].astype(np.float32)
            _elv = lidar["lidar_pts"][:, 1].astype(np.float32)
            _x0, _z0 = float(_x.min()), float(_z.min())
            _col = ((_x - _x0) / _cs).astype(np.int32)
            _row = ((_z - _z0) / _cs).astype(np.int32)
            _W, _H = int(_col.max()) + 1, int(_row.max()) + 1
            _flat = np.full(_H * _W, -np.inf, dtype=np.float32)
            np.maximum.at(_flat, _row * _W + _col, _elv)
            _dem_li = _flat.reshape(_H, _W).astype(float)
            _dem_li[_dem_li == -np.inf] = np.nan
            _li_shorter = min(_H, _W)
            _li_radius  = _li_shorter / 4.0
            _li_cr, _li_cc = _H / 2.0, _W / 2.0
            _rr_li, _cc_li = np.mgrid[0:_H, 0:_W]
            _circle_li = ((_rr_li - _li_cr)**2 + (_cc_li - _li_cc)**2) <= _li_radius**2

            # --- build GeoTIFF DEM once ---
            _ds  = _gdal.Open(dem_path)
            _gt  = _ds.GetGeoTransform()
            _W_d, _H_d = _ds.RasterXSize, _ds.RasterYSize
            _band   = _ds.GetRasterBand(1)
            _nodata = _band.GetNoDataValue()
            _dem_gt = _band.ReadAsArray().astype(float)
            if _nodata is not None:
                _dem_gt[_dem_gt == _nodata] = np.nan
            _cx_gt, _cy_gt = float(_gt[1]), abs(float(_gt[5]))  # pixel size x, y
            _phys_h = _H_d * _cy_gt
            _phys_w = _W_d * _cx_gt
            _gt_shorter = min(_phys_h, _phys_w)
            _gt_radius_m = _gt_shorter / 4.0
            _gt_rr = _gt_radius_m / _cy_gt
            _gt_rc = _gt_radius_m / _cx_gt
            _gt_cr, _gt_cc = _H_d / 2.0, _W_d / 2.0
            _rr_gt, _cc_g_gt = np.mgrid[0:_H_d, 0:_W_d]
            _circle_gt = (((_rr_gt - _gt_cr) / _gt_rr)**2 +
                          ((_cc_g_gt - _gt_cc) / _gt_rc)**2) <= 1.0
            _dem_utm_x0    = _gt[0]
            _dem_utm_y_top = _gt[3]
            _dem_utm_y_bot = _gt[3] + _gt[5] * _H_d
            _ply_x0, _ply_y0 = render_world_bbox[:2]
            _off_x = _dem_utm_x0 - _ply_x0
            _off_y = _dem_utm_y_bot - _ply_y0

            def _pca2(pts):
                center = pts.mean(axis=0)
                cov    = np.cov(pts.T)
                ev, evec = np.linalg.eigh(cov)
                main  = evec[:, -1]
                return center, float(np.degrees(np.arctan2(main[1], main[0])))

            def _make_tfm(ang_li, cx_li, ang_pl, cx_pl, xz_poly, ply_render, render_bbox):
                import math
                def _rot_translate(pts, angle_deg, cx_src, cx_dst, scale=1.0):
                    a = math.radians(angle_deg)
                    R = np.array([[math.cos(a), -math.sin(a)],
                                  [math.sin(a),  math.cos(a)]])
                    return (pts - cx_src) @ R.T * scale + cx_dst

                rH, rW = ply_render.shape[:2]
                rx0, ry0, rx1, ry1 = render_bbox
                for rot in [ang_pl - ang_li, ang_pl - ang_li + 180.0]:
                    poly_w = _rot_translate(xz_poly, rot, cx_li, cx_pl)
                    px = ((poly_w[:, 0] - rx0) / (rx1 - rx0) * rW).astype(int)
                    py = ((ry1 - poly_w[:, 1]) / (ry1 - ry0) * rH).astype(int)
                    if (px.min() >= 0 and px.max() < rW and
                            py.min() >= 0 and py.max() < rH):
                        return rot, lambda p, r=rot, s=cx_li, d=cx_pl: _rot_translate(p, r, s, d)
                rot = ang_pl - ang_li
                return rot, lambda p, r=rot, s=cx_li, d=cx_pl: _rot_translate(p, r, s, d)

            for pool_k in [5, 15, 30, 60]:
                tag = f"exp_minpool_k{pool_k:02d}"
                pct = 60
                print(f"\n  [MinPool k={pool_k}px] pool={pool_k*_cs:.2f}m LiDAR "
                      f"/ pool={pool_k*_cx_gt:.2f}m GeoTIFF, pct={pct}")
                mp_folder = os.path.join(output_dir, tag)
                try:
                    # --- LiDAR side ---
                    dem_li_p = minimum_filter(_dem_li, size=pool_k)
                    dem_li_p[np.isnan(_dem_li)] = np.nan  # restore NaN gaps
                    in_li = (~np.isnan(dem_li_p)) & _circle_li
                    thresh_li = float(np.percentile(dem_li_p[in_li], pct))
                    wr_li, wc_li = np.where(in_li & (dem_li_p <= thresh_li))
                    pts_li = np.stack([_x0 + wc_li * _cs, _z0 + wr_li * _cs], axis=1)
                    print(f"    LiDAR: {len(pts_li)} cells selected (thresh={thresh_li:.3f})")

                    dmin_li = float(np.nanmin(_dem_li)); dmax_li = float(np.nanmax(_dem_li))
                    img8_li = (((_dem_li - dmin_li) / max(dmax_li - dmin_li, 1e-6)) * 255)
                    img8_li = np.nan_to_num(img8_li, nan=0).astype(np.uint8)
                    dbg_li = cv2.cvtColor(img8_li, cv2.COLOR_GRAY2BGR)
                    dbg_li[wr_li, wc_li] = (0, 0, 255)
                    cv2.circle(dbg_li, (int(_li_cc), int(_li_cr)), int(_li_radius), (0,255,0), 2)

                    # --- GeoTIFF side ---
                    dem_gt_p = minimum_filter(_dem_gt, size=pool_k)
                    dem_gt_p[np.isnan(_dem_gt)] = np.nan
                    in_gt = (~np.isnan(dem_gt_p)) & _circle_gt
                    thresh_gt = float(np.percentile(dem_gt_p[in_gt], pct))
                    wr_gt, wc_gt = np.where(in_gt & (dem_gt_p <= thresh_gt))
                    utm_x  = _dem_utm_x0    + wc_gt * _gt[1]
                    utm_y  = _dem_utm_y_top + wr_gt * _gt[5]
                    pts_gt = np.stack([utm_x - _off_x, utm_y - _off_y], axis=1)
                    print(f"    GeoTIFF: {len(pts_gt)} cells selected (thresh={thresh_gt:.3f}m)")

                    dmin_gt = float(np.nanmin(_dem_gt)); dmax_gt = float(np.nanmax(_dem_gt))
                    img8_gt = (((_dem_gt - dmin_gt) / max(dmax_gt - dmin_gt, 1e-6)) * 255)
                    img8_gt = np.nan_to_num(img8_gt, nan=0).astype(np.uint8)
                    dbg_gt = cv2.cvtColor(img8_gt, cv2.COLOR_GRAY2BGR)
                    dbg_gt[wr_gt, wc_gt] = (0, 0, 255)
                    cv2.ellipse(dbg_gt, (int(_gt_cc), int(_gt_cr)),
                                (int(_gt_rc), int(_gt_rr)), 0, 0, 360, (0,255,0), 2)

                    if len(pts_li) < 10 or len(pts_gt) < 10:
                        raise RuntimeError("Too few points after min-pool selection")

                    # --- PCA + registration ---
                    cx_li, ang_li = _pca2(pts_li)
                    cx_pl, ang_pl = _pca2(pts_gt)
                    rot, tfm = _make_tfm(ang_li, cx_li, ang_pl, cx_pl,
                                         lidar["xz_polygon"], render_img, render_world_bbox)
                    yellow_w = tfm(lidar["xz_polygon"])
                    print(f"    Rotation={rot:.1f}°  polygon: "
                          f"X=[{yellow_w[:,0].min():.3f},{yellow_w[:,0].max():.3f}] "
                          f"Y=[{yellow_w[:,1].min():.3f},{yellow_w[:,1].max():.3f}]")

                    # --- registration overlay on PLY render ---
                    reg_img = render_img.copy()
                    rH, rW = render_img.shape[:2]
                    rx0, ry0, rx1, ry1 = render_world_bbox
                    px = np.clip(((yellow_w[:,0]-rx0)/(rx1-rx0)*rW).astype(int), 0, rW-1)
                    py = np.clip(((ry1-yellow_w[:,1])/(ry1-ry0)*rH).astype(int), 0, rH-1)
                    pts_draw = np.stack([px, py], axis=1).reshape(-1, 1, 2)
                    cv2.polylines(reg_img, [pts_draw], True, (0, 0, 255), 3)

                    _save_imgs(mp_folder, {
                        f"SU{su}_lidar_dem.png":      dbg_li,
                        f"SU{su}_ply_dem.png":        dbg_gt,
                        f"SU{su}_registration.png":   reg_img,
                        f"SU{su}_snip_reference.png": _snip_reference(
                                                          render_img, render_world_bbox,
                                                          yellow_w),
                    })
                except Exception as e:
                    import traceback
                    print(f"  MinPool k={pool_k} failed: {e}")
                    traceback.print_exc()
        else:
            print("  [MinPool] Skipped (no GeoTIFF DEM)")

        # ------------------------------------------------------------------
        # Experiment 2: 2-D ICP refinement (kept for reference)
        # ------------------------------------------------------------------
        print(f"\n  [Exp 2] 2-D ICP refinement (RGB PCA initial + horizontal ICP)")
        icp_folder = os.path.join(output_dir, "exp_icp")
        try:
            tfm_i, reg_i, note_i, icp_debug = \
                auto_snip_lidar.register_lidar_to_ply_world_icp(
                    lidar["lidar_pts"], lidar["lidar_xz_bbox"],
                    top_cloud, render_world_bbox,
                    lidar["lidar_render"], render_img,
                    lidar["xz_polygon"],
                )
            yellow_world_i = tfm_i(lidar["xz_polygon"])
            print(f"  2-D ICP polygon: "
                  f"X=[{yellow_world_i[:,0].min():.3f},{yellow_world_i[:,0].max():.3f}] "
                  f"Y=[{yellow_world_i[:,1].min():.3f},{yellow_world_i[:,1].max():.3f}]")
            _save_imgs(icp_folder, {
                f"SU{su}_initial_pca.png":      icp_debug.get("initial_pca_img"),
                f"SU{su}_registration_icp.png": reg_i,
                f"SU{su}_snip_reference.png":   _snip_reference(
                                                    render_img, render_world_bbox,
                                                    yellow_world_i),
            })
        except Exception as e:
            print(f"  2-D ICP experiment failed: {e}")

        # ------------------------------------------------------------------
        # Experiment 3: 3-D ICP refinement
        # ------------------------------------------------------------------
        print(f"\n  [Exp 3] 3-D ICP refinement (RGB PCA initial + full 3-D ICP)")
        icp3d_folder = os.path.join(output_dir, "exp_icp_3d")
        try:
            tfm_3d, reg_3d, note_3d, icp3d_debug = \
                auto_snip_lidar.register_lidar_to_ply_world_icp_3d(
                    lidar["lidar_pts"], lidar["lidar_xz_bbox"],
                    top_cloud, render_world_bbox,
                    lidar["lidar_render"], render_img,
                    lidar["xz_polygon"],
                )
            yellow_world_3d = tfm_3d(lidar["xz_polygon"])
            print(f"  3-D ICP polygon: "
                  f"X=[{yellow_world_3d[:,0].min():.3f},{yellow_world_3d[:,0].max():.3f}] "
                  f"Y=[{yellow_world_3d[:,1].min():.3f},{yellow_world_3d[:,1].max():.3f}]")
            _save_imgs(icp3d_folder, {
                f"SU{su}_initial_pca.png":          icp3d_debug.get("initial_pca_img"),
                f"SU{su}_registration_icp_3d.png":  reg_3d,
                f"SU{su}_snip_reference.png":        _snip_reference(
                                                        render_img, render_world_bbox,
                                                        yellow_world_3d),
            })
        except Exception as e:
            print(f"  3-D ICP experiment failed: {e}")

print("\nExperiments complete.")
