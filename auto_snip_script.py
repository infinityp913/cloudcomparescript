import cloudComPy as cc
import cv2
import numpy as np
import json
import os
import sys
import glob

from pre_snip_script import DATA_DIR, find_mesh_by_pgram_job, INPUT_MESH_PATH
import auto_snip_lidar

cc.initCC()

json_filepath = sys.argv[1] if len(sys.argv) > 1 else "example-17000.json"
json_id = os.path.splitext(os.path.basename(json_filepath))[0]  # e.g. "example-20002"

with open(json_filepath, "r") as f:
    job_data = json.load(f)


# ---------------------------------------------------------------------------
# Annotation image parsing
# ---------------------------------------------------------------------------

def parse_su_number(annotation_path: str) -> str:
    base = os.path.basename(annotation_path)
    parts = base.split("_")
    for i, part in enumerate(parts):
        if part.upper() == "SU" and i + 1 < len(parts):
            return parts[i + 1]
    raise ValueError(f"Cannot parse SU number from filename: {base}")


def extract_polygon_for_color(img_bgr: np.ndarray, color: str) -> np.ndarray | None:
    """
    Returns convex hull of the dominant connected cluster of the given annotation
    color as (N, 2) array of (x_px, y_px), or None if not found.
    Pure annotation colors have S > 150; muted terrain colors have S < 100.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "yellow": [(np.array([20, 150, 100]), np.array([40, 255, 255]))],
        "red":    [(np.array([0,  150, 100]), np.array([10, 255, 255])),
                   (np.array([165, 150, 100]), np.array([180, 255, 255]))],
        "blue":   [(np.array([90, 150, 100]), np.array([130, 255, 255]))],
    }
    if color not in color_ranges:
        raise ValueError(f"Unknown color: {color}")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in color_ranges[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == largest_label)
    if len(xs) < 5:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    hull_idx = cv2.convexHull(points.reshape(-1, 1, 2), returnPoints=False)
    hull = points[hull_idx.flatten()]
    print(f"    {color}: {len(xs)} cluster px → hull {len(hull)} vertices "
          f"(x={xs.min()}-{xs.max()}, y={ys.min()}-{ys.max()})")
    return hull.astype(float)


# ---------------------------------------------------------------------------
# Top-down render of the point cloud
# ---------------------------------------------------------------------------

def render_topdown_image(cloud, resolution: float = 0.01) -> tuple:
    """
    Render a top-down XY projection of the cloud as a BGR image.
    Returns (image_bgr, (x0, y0, x1, y1)) where the bbox is in world coords.
    resolution: metres per pixel.
    """
    coords = cloud.toNpArrayCopy()  # (N, 3)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    x0, y0, x1, y1 = float(x.min()), float(y.min()), float(x.max()), float(y.max())
    W = int(np.ceil((x1 - x0) / resolution)) + 1
    H = int(np.ceil((y1 - y0) / resolution)) + 1
    print(f"  Render size: {W}x{H} px at {resolution*100:.0f} cm/px")

    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Pixel positions (Y flipped: world Y up -> image row down)
    px = np.clip(((x - x0) / resolution).astype(int), 0, W - 1)
    py = np.clip((H - 1 - (y - y0) / resolution).astype(int), 0, H - 1)

    # Sort by Z so topmost points overwrite lower ones
    order = np.argsort(z)
    px_s, py_s = px[order], py[order]

    if cloud.hasColors():
        try:
            rgba = cloud.colorsToNpArrayCopy()  # (N, 4) uint8 RGBA
            bgr = rgba[order, :3][:, ::-1]      # RGBA -> BGR
            img[py_s, px_s] = bgr
            print("  Render: using RGB colors from cloud")
        except Exception as e:
            print(f"  Render: color extraction failed ({e}), using height map")
    if img.max() == 0:
        z_s = z[order]
        z_norm = ((z_s - z_s.min()) / max(z_s.max() - z_s.min(), 1e-6) * 255).astype(np.uint8)
        gray = np.stack([z_norm, z_norm, z_norm], axis=1)
        img[py_s, px_s] = gray
        print("  Render: using height-map colors")

    # Dilate to fill sparse gaps
    img = cv2.dilate(img, np.ones((7, 7), np.uint8))

    return img, (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# PCA-based similarity transform: annotation image -> rendered cloud image
#
# Feature matching (ORB/SIFT) fails because the annotation and render have
# a large rotation difference and different visual characteristics (orthographic
# render vs oblique photo). Instead we align the geometry:
#   - PCA on the red+blue annotation outline pixels -> trench orientation & center
#   - PCA on the non-black render pixels -> same for render footprint
#   - Build similarity transform (rotation + scale + translation) from these.
# ---------------------------------------------------------------------------

def _get_rb_pixels(annotation_bgr: np.ndarray) -> np.ndarray:
    """Return (N,2) (x,y) pixel coords of all red+blue annotation outline pixels."""
    hsv = cv2.cvtColor(annotation_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in [
        (np.array([0,   150, 100]), np.array([10,  255, 255])),
        (np.array([165, 150, 100]), np.array([180, 255, 255])),
        (np.array([90,  150, 100]), np.array([130, 255, 255])),
    ]:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    return np.column_stack([xs, ys]).astype(float)


def _get_render_pixels(render_bgr: np.ndarray) -> np.ndarray:
    """Return (N,2) (x,y) pixel coords of all non-black render pixels."""
    gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 10)
    return np.column_stack([xs, ys]).astype(float)


def _pca(pts: np.ndarray) -> tuple:
    """PCA: returns (center_xy, main_angle_deg, std_main, std_perp)."""
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    main = eigenvectors[:, -1]
    angle = np.degrees(np.arctan2(main[1], main[0]))
    return center, angle, np.sqrt(eigenvalues[-1]), np.sqrt(eigenvalues[0])


def find_annotation_to_world_transform(
    annotation_bgr: np.ndarray,
    render_bgr: np.ndarray,
    render_world_bbox: tuple,
    yellow_hull_px: np.ndarray,
) -> tuple:
    """
    PCA-based similarity transform: annotation pixels -> world XY.
    Aligns the red+blue trench outline (annotation) to the render footprint
    using their principal axes (center, orientation, extent).
    Tries both 180-degree orientations; picks the one where the yellow polygon
    projects within the render image bounds.

    Returns (transform_fn, debug_ann_img).
    transform_fn: (N,2) annotation pixels -> (N,2) world XY.
    """
    ann_pts = _get_rb_pixels(annotation_bgr)
    ren_pts = _get_render_pixels(render_bgr)

    if len(ann_pts) < 10:
        raise RuntimeError("PCA: too few red+blue pixels in annotation")
    if len(ren_pts) < 10:
        raise RuntimeError("PCA: render image appears empty")

    cx_ann, ang_ann, std_main_ann, std_perp_ann = _pca(ann_pts)
    cx_ren, ang_ren, std_main_ren, std_perp_ren = _pca(ren_pts)

    scale = (std_main_ren / std_main_ann + std_perp_ren / std_perp_ann) / 2
    rH, rW = render_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = render_world_bbox

    print(f"  PCA ann:    center=({cx_ann[0]:.0f},{cx_ann[1]:.0f}) "
          f"angle={ang_ann:.1f} main={std_main_ann:.0f}px perp={std_perp_ann:.0f}px")
    print(f"  PCA render: center=({cx_ren[0]:.0f},{cx_ren[1]:.0f}) "
          f"angle={ang_ren:.1f} main={std_main_ren:.0f}px perp={std_perp_ren:.0f}px")
    print(f"  PCA scale: {scale:.4f}")

    def _make_M(rot_deg: float) -> np.ndarray:
        rot_rad = np.radians(rot_deg)
        c, s = np.cos(rot_rad), np.sin(rot_rad)
        tx = cx_ren[0] - scale * (c * cx_ann[0] - s * cx_ann[1])
        ty = cx_ren[1] - scale * (s * cx_ann[0] + c * cx_ann[1])
        return np.array([[scale * c, -scale * s, tx],
                         [scale * s,  scale * c, ty],
                         [0, 0, 1]])

    def _apply_M(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M @ h.T).T[:, :2]

    def _within_render(rpts: np.ndarray, margin: float = 0.15) -> bool:
        return (rpts[:, 0].min() > -rW * margin and
                rpts[:, 0].max() <  rW * (1 + margin) and
                rpts[:, 1].min() > -rH * margin and
                rpts[:, 1].max() <  rH * (1 + margin))

    rot_deg = ang_ren - ang_ann
    chosen_M = None
    for rotation in [rot_deg, rot_deg + 180]:
        M = _make_M(rotation)
        if _within_render(_apply_M(M, yellow_hull_px)):
            print(f"  PCA rotation={rotation:.1f}deg -> yellow polygon within render bounds")
            chosen_M = M
            break

    if chosen_M is None:
        chosen_M = _make_M(rot_deg)
        print(f"  PCA rotation={rot_deg:.1f}deg (fallback)")

    def transform(annotation_pts: np.ndarray) -> np.ndarray:
        render_pts = _apply_M(chosen_M, annotation_pts)
        world = np.empty_like(render_pts, dtype=float)
        world[:, 0] = rx0 + render_pts[:, 0] * (rx1 - rx0) / rW
        world[:, 1] = ry1 - render_pts[:, 1] * (ry1 - ry0) / rH
        return world

    # Debug: annotation with yellow polygon drawn
    debug = annotation_bgr.copy()
    pts = yellow_hull_px.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    return transform, debug


# ---------------------------------------------------------------------------
# Point cloud cropping
# ---------------------------------------------------------------------------

def crop_cloud_by_polygon_2d(cloud, polygon_world):
    """
    Return a new cloud containing only points whose (x, y) fall inside
    any of the given 2D polygons in world coords.
    polygon_world may be a single (N, 2) array or a list of (N, 2) arrays.
    Uses a scalar field + cc.filterBySFValue (idiomatic CloudComPy).
    """
    from matplotlib.path import Path

    if isinstance(polygon_world, np.ndarray):
        polygon_world = [polygon_world]

    coords = cloud.toNpArrayCopy()
    mask = np.zeros(len(coords), dtype=bool)
    for poly in polygon_world:
        mask |= Path(poly).contains_points(coords[:, :2])

    if not mask.any():
        print(f"  Warning: polygon crop produced 0 points for '{cloud.getName()}'")
        return None

    sf_name = "__polygon_mask__"
    sf_idx = cloud.addScalarField(sf_name)
    sf = cloud.getScalarField(sf_idx)
    sf.fromNpArrayCopy(np.where(mask, 1.0, 0.0).astype(np.float32))
    cloud.setCurrentOutScalarField(sf_idx)

    filtered = cc.filterBySFValue(0.5, 2.0, cloud)
    cloud.deleteScalarField(sf_idx)

    if filtered is None or filtered.size() == 0:
        print(f"  Warning: filterBySFValue returned empty cloud for '{cloud.getName()}'")
        return None

    filtered.setName(cloud.getName() + "_cropped")
    return filtered


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_presnip_bins(top_id: str, bottom_id: str) -> tuple[str, str]:
    search_dir = os.path.join(DATA_DIR, top_id)
    bins = glob.glob(os.path.join(search_dir, "*.bin"))
    top_bin = bottom_bin = None
    for b in bins:
        name = os.path.basename(b).lower()
        if "top_with_dist" in name:
            top_bin = b
        elif "bottom_with_dist" in name:
            bottom_bin = b
    if top_bin is None or bottom_bin is None:
        raise FileNotFoundError(
            f"Could not find pre-snipped .bin pair in {search_dir}. "
            "Run pre_snip_script.py first."
        )
    return top_bin, bottom_bin


# ---------------------------------------------------------------------------
# Save cleaned cloud
# ---------------------------------------------------------------------------

def save_cleaned_cloud(cloud, output_dir: str, base_name: str, su_number: str, is_top: bool) -> str:
    suffix = "_top" if is_top else ""
    filename = f"{base_name}_cleaned_su_{su_number}{suffix}.bin"
    save_path = os.path.join(output_dir, filename)
    res = cc.SavePointCloud(cloud, save_path)
    if res != 0:
        print(f"  Warning: SavePointCloud returned {res} for {filename}")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting auto-snip processing...")

    for job in job_data:
        top_job    = job["top"]
        bottom_job = job["bottom"]
        annotations = job.get("annotations", [])

        if not annotations:
            print(f"No annotations for top={top_job} / bottom={bottom_job}, skipping.")
            continue

        top_id    = find_mesh_by_pgram_job(top_job,    INPUT_MESH_PATH)
        bottom_id = find_mesh_by_pgram_job(bottom_job, INPUT_MESH_PATH)

        if top_id is None or bottom_id is None:
            print(f"Could not find PLY for top={top_job} or bottom={bottom_job}, skipping.")
            continue

        print(f"\nProcessing pair: top={top_id}  bottom={bottom_id}")

        try:
            top_bin, bottom_bin = find_presnip_bins(top_id, bottom_id)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

        print("  Loading pre-snipped clouds...")
        top_cloud    = cc.loadPointCloud(top_bin)
        bottom_cloud = cc.loadPointCloud(bottom_bin)
        print(f"  Top: {top_cloud.size()} pts  Bottom: {bottom_cloud.size()} pts")

        # Render top-down image from the top cloud (shared across all SUs in this pair)
        print("  Rendering top-down image from top cloud...")
        render_img, render_world_bbox = render_topdown_image(top_cloud, resolution=0.01)
        output_dir = os.path.join(DATA_DIR, json_id)
        os.makedirs(output_dir, exist_ok=True)
        render_path = os.path.join(output_dir, "debug_topdown_render.png")
        cv2.imwrite(render_path, render_img)
        print(f"  Render saved: {render_path}")

        for annotation_path in annotations:
            ext = os.path.splitext(annotation_path)[1].lower()
            print(f"\n  Processing annotation: {annotation_path}")

            # ------------------------------------------------------------------
            # Branch A: USDZ (LiDAR scan with physically-painted yellow annotation)
            # ------------------------------------------------------------------
            if ext == '.usdz':
                try:
                    lidar = auto_snip_lidar.process_usdz(annotation_path)
                except Exception as e:
                    print(f"  Error processing USDZ: {e}")
                    continue

                su_number = lidar["su_name"]
                print(f"  SU name from USDZ: {su_number}")

                # Save LiDAR debug images
                cv2.imwrite(os.path.join(output_dir, "debug_lidar_render.png"),
                            lidar["lidar_render"])
                auto_snip_lidar.save_lidar_debug(
                    lidar["lidar_render"], lidar["xz_polygon"],
                    lidar["lidar_xz_bbox"],
                    os.path.join(output_dir, f"debug_SU{su_number}_lidar_yellow.png"),
                )

                # ------------------------------------------------------------------
                # Registration cascade:
                #   1. Hybrid: RGB PCA rotation + DEM floor centroid translation
                #   2. Fallback: plain RGB footprint PCA
                # ICP comparison always runs afterwards (result logged but not
                # used for the crop so we can compare the two outputs).
                # ------------------------------------------------------------------
                dem_path = os.path.join(DATA_DIR, "DEMs", f"{top_id}_dem.tif")

                # Registration strategy:
                # 1. PreRotAKAZE — reliable when the scene has texture correspondence
                #    (≥20 RANSAC inliers).  Fails gracefully when the modality gap is
                #    too large (LiDAR interior walls vs PLY top-down view).
                # 2. PCA-Chamfer — fallback when AKAZE can't find enough matches.
                #    Locks PCA rotation and chamfer-searches translation only.
                # 3. RGB PCA — last resort.
                AKAZE_MIN_INLIERS = 20
                _reg_args = (lidar["lidar_render"], lidar["lidar_xz_bbox"],
                             render_img, render_world_bbox, lidar["xz_polygon"])
                transform = debug_reg = reg_note = reg_debug = None
                try:
                    transform, debug_reg, reg_note, reg_debug = \
                        auto_snip_lidar.register_lidar_to_ply_world_prerot_akaze(*_reg_args)
                    if reg_debug.get("inliers", 0) < AKAZE_MIN_INLIERS:
                        print(f"  AKAZE only {reg_debug['inliers']} inliers "
                              f"(< {AKAZE_MIN_INLIERS}) — falling back to PCA-Chamfer")
                        transform = None
                except RuntimeError as e:
                    print(f"  AKAZE failed: {e} — falling back to PCA-Chamfer")

                if transform is None:
                    try:
                        transform, debug_reg, reg_note, reg_debug = \
                            auto_snip_lidar.register_lidar_to_ply_world_pca_chamfer(*_reg_args)
                    except RuntimeError as e:
                        print(f"  PCA-Chamfer failed: {e} — falling back to RGB PCA")
                        try:
                            transform, debug_reg, reg_note, reg_debug = \
                                auto_snip_lidar.register_lidar_to_ply_world(*_reg_args)
                        except RuntimeError as e2:
                            print(f"  Registration failed: {e2}")
                            continue

                reg_path = os.path.join(output_dir, f"debug_SU{su_number}_registration.png")
                cv2.imwrite(reg_path, debug_reg)
                print(f"  Registration ({reg_note}): {reg_path}")
                for key, img_arr in reg_debug.items():
                    if not isinstance(img_arr, np.ndarray):
                        continue
                    cv2.imwrite(
                        os.path.join(output_dir, f"debug_SU{su_number}_{key}.png"), img_arr)
                    print(f"  Saved: debug_SU{su_number}_{key}.png")

                # --- experimental registration methods (compare side-by-side) ---
                # Render-only methods: uniform (lidar_render, bbox, ply, bbox, poly) sig
                _exp_methods = [
                    ("phase_corr",    auto_snip_lidar.register_lidar_to_ply_world_phase_corr),
                    ("prerot_akaze",  auto_snip_lidar.register_lidar_to_ply_world_prerot_akaze),
                    ("annot_bndry",   auto_snip_lidar.register_lidar_to_ply_world_annot_boundary),
                    ("dist_pca",      auto_snip_lidar.register_lidar_to_ply_world_dist_pca),
                    ("pca_chamfer",   auto_snip_lidar.register_lidar_to_ply_world_pca_chamfer),
                    ("mutual_info",   auto_snip_lidar.register_lidar_to_ply_world_mutual_info),
                ]
                for _mname, _mfn in _exp_methods:
                    print(f"  [Exp] Running {_mname} ...")
                    try:
                        _, _, _mnote, _mdbg = _mfn(
                            lidar["lidar_render"], lidar["lidar_xz_bbox"],
                            render_img, render_world_bbox,
                            lidar["xz_polygon"],
                        )
                        for _key, _img in _mdbg.items():
                            _p = os.path.join(output_dir,
                                              f"debug_SU{su_number}_{_mname}_{_key}.png")
                            cv2.imwrite(_p, _img)
                        print(f"  [Exp:{_mname}] {_mnote}")
                    except Exception as _merr:
                        print(f"  [Exp:{_mname}] Failed: {_merr}")

                # DEM-ridge method: needs lidar_pts + dem_path (separate signature)
                if os.path.exists(dem_path):
                    print(f"  [Exp] Running dem_ridge ...")
                    try:
                        _, _, _rnote, _rdbg = \
                            auto_snip_lidar.register_lidar_to_ply_world_dem_ridge(
                                lidar["lidar_pts"], lidar["lidar_xz_bbox"],
                                dem_path, render_world_bbox,
                                lidar["lidar_render"], render_img,
                                lidar["xz_polygon"],
                            )
                        for _key, _img in _rdbg.items():
                            _p = os.path.join(output_dir,
                                              f"debug_SU{su_number}_dem_ridge_{_key}.png")
                            cv2.imwrite(_p, _img)
                        print(f"  [Exp:dem_ridge] {_rnote}")
                    except Exception as _rerr:
                        print(f"  [Exp:dem_ridge] Failed: {_rerr}")

                # Transform all polygon clusters; crop is the union of all of them
                yellow_worlds = [transform(poly) for poly in lidar["xz_polygons"]]
                yellow_world  = yellow_worlds[0]  # largest, for display/reference
                if len(yellow_worlds) > 1:
                    print(f"  Multi-polygon: {len(yellow_worlds)} regions detected")

            # ------------------------------------------------------------------
            # Branch B: PNG annotation (top-down photo with painted color lines)
            # ------------------------------------------------------------------
            elif ext == '.png':
                su_number = parse_su_number(annotation_path)
                print(f"  SU number: {su_number}")

                img_bgr = cv2.imread(annotation_path)
                if img_bgr is None:
                    print(f"  Error: cannot read {annotation_path}")
                    continue

                yellow_px = extract_polygon_for_color(img_bgr, "yellow")
                if yellow_px is None:
                    print(f"  Error: no yellow polygon found in {annotation_path}")
                    continue
                print(f"  Yellow polygon: {len(yellow_px)} hull vertices")

                try:
                    transform, debug_ann = find_annotation_to_world_transform(
                        img_bgr, render_img, render_world_bbox, yellow_px
                    )
                except RuntimeError as e:
                    print(f"  Error: {e}")
                    continue

                cv2.imwrite(os.path.join(output_dir, f"debug_SU{su_number}_annotation.png"), debug_ann)
                yellow_world  = transform(yellow_px)
                yellow_worlds = [yellow_world]

            else:
                print(f"  Unsupported annotation format '{ext}', skipping.")
                continue

            # ------------------------------------------------------------------
            # Common: print world extent, save snip reference, crop, save
            # ------------------------------------------------------------------
            print(f"  Yellow polygon world extent: "
                  f"X=[{yellow_world[:,0].min():.3f}, {yellow_world[:,0].max():.3f}]  "
                  f"Y=[{yellow_world[:,1].min():.3f}, {yellow_world[:,1].max():.3f}]")

            # Snip reference: full render (left) | dimmed render + all polygons highlighted (right)
            rH, rW = render_img.shape[:2]
            rx0, ry0, rx1, ry1 = render_world_bbox

            def _world_to_render_px(yw):
                px = np.empty((len(yw), 2), dtype=int)
                px[:, 0] = np.clip(((yw[:, 0] - rx0) / (rx1 - rx0) * rW).astype(int), 0, rW-1)
                px[:, 1] = np.clip(((ry1 - yw[:, 1]) / (ry1 - ry0) * rH).astype(int), 0, rH-1)
                return px

            highlight = render_img.copy()
            su_mask = np.zeros((rH, rW), dtype=np.uint8)
            for yw in yellow_worlds:
                rpx = _world_to_render_px(yw)
                cv2.fillPoly(su_mask, [rpx.reshape(-1, 1, 2)], 255)
            highlight[su_mask == 0] = (highlight[su_mask == 0].astype(float) * 0.35).astype(np.uint8)
            for yw in yellow_worlds:
                rpx = _world_to_render_px(yw)
                cv2.polylines(highlight, [rpx.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)

            snip_ref = np.hstack([render_img, highlight])
            ref_path = os.path.join(output_dir, f"debug_SU{su_number}_snip_reference.png")
            cv2.imwrite(ref_path, snip_ref)
            print(f"  Snip reference saved: {ref_path}")

            # Crop both clouds to the union of all yellow polygons
            print("  Cropping top cloud...")
            top_cropped = crop_cloud_by_polygon_2d(top_cloud, yellow_worlds)
            print("  Cropping bottom cloud...")
            bottom_cropped = crop_cloud_by_polygon_2d(bottom_cloud, yellow_worlds)

            if top_cropped is None or bottom_cropped is None:
                print(f"  Error: crop produced empty cloud for SU {su_number}, skipping.")
                continue

            print(f"  Top cropped: {top_cropped.size()} pts  "
                  f"Bottom cropped: {bottom_cropped.size()} pts")

            top_path = save_cleaned_cloud(top_cropped,    output_dir, top_id,    su_number, is_top=False)
            bot_path = save_cleaned_cloud(bottom_cropped, output_dir, bottom_id, su_number, is_top=False)
            print(f"  Saved: {os.path.basename(top_path)}")
            print(f"  Saved: {os.path.basename(bot_path)}")

            poly_path = os.path.join(output_dir, f"su_{su_number}_polygon.npy")
            np.save(poly_path, yellow_world)
            print(f"  Saved polygon: su_{su_number}_polygon.npy")

    print("\nAuto-snip complete.")
