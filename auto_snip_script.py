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


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_su_number(annotation_path: str) -> str | None:
    """Return SU number from a filename containing '_SU_<number>', or None."""
    base = os.path.basename(annotation_path)
    parts = base.split("_")
    for i, part in enumerate(parts):
        if part.upper() == "SU" and i + 1 < len(parts):
            return parts[i + 1]
    return None


# ---------------------------------------------------------------------------
# Top-down render of the point cloud
# ---------------------------------------------------------------------------

def render_topdown_image(cloud, resolution: float = 0.01) -> tuple:
    """
    Render a top-down XY projection of the cloud as a BGR image.
    Returns (image_bgr, (x0, y0, x1, y1)) where bbox is in world coords.
    """
    coords = cloud.toNpArrayCopy()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    x0, y0, x1, y1 = float(x.min()), float(y.min()), float(x.max()), float(y.max())
    W = int(np.ceil((x1 - x0) / resolution)) + 1
    H = int(np.ceil((y1 - y0) / resolution)) + 1
    print(f"  Render size: {W}x{H} px at {resolution*100:.0f} cm/px")

    img = np.zeros((H, W, 3), dtype=np.uint8)
    px = np.clip(((x - x0) / resolution).astype(int), 0, W - 1)
    py = np.clip((H - 1 - (y - y0) / resolution).astype(int), 0, H - 1)

    order = np.argsort(z)
    px_s, py_s = px[order], py[order]

    if cloud.hasColors():
        try:
            rgba = cloud.colorsToNpArrayCopy()
            bgr = rgba[order, :3][:, ::-1]
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

    img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    return img, (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Point cloud cropping
# ---------------------------------------------------------------------------

def crop_cloud_by_polygon_2d(cloud, polygon_world):
    """
    Return a new cloud containing only points whose (x, y) fall inside
    any of the given 2D polygons in world coords.
    polygon_world may be a single (N, 2) array or a list of (N, 2) arrays.
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
# File discovery + save
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


def save_cleaned_cloud(cloud, output_dir: str, base_name: str, su_number: str, is_top: bool) -> str:
    suffix = "_top" if is_top else ""
    filename = f"{base_name}_cleaned_su_{su_number}{suffix}.bin"
    save_path = os.path.join(output_dir, filename)
    res = cc.SavePointCloud(cloud, save_path)
    if res != 0:
        print(f"  Warning: SavePointCloud returned {res} for {filename}")
    return save_path


# ---------------------------------------------------------------------------
# Mode 1: Autosnip — LiDAR USDZ registration
# ---------------------------------------------------------------------------

def run_autosnip(lidar: dict, render_img: np.ndarray, render_world_bbox: tuple,
                 output_dir: str, su_number: str) -> tuple:
    """
    Register a LiDAR USDZ scan to the PLY world frame using 5 math methods.
    rgb_pca is the default used for cropping; all 5 generate debug composites.
    Returns (transform_fn, yellow_worlds) from rgb_pca, or (None, None) on failure.

    API methods (Claude Vision, OpenRouter) are commented out but available
    in auto_snip_lidar.py for future use.
    """
    METHODS = [
        ("rgb_pca",      auto_snip_lidar.register_lidar_to_ply_world),        # DEFAULT
        ("dist_pca",     auto_snip_lidar.register_lidar_to_ply_world_dist_pca),
        ("phase_corr",   auto_snip_lidar.register_lidar_to_ply_world_phase_corr),
        ("prerot_akaze", auto_snip_lidar.register_lidar_to_ply_world_prerot_akaze),
        ("pca_chamfer",  auto_snip_lidar.register_lidar_to_ply_world_pca_chamfer),
    ]
    _args = (lidar["lidar_render"], lidar["lidar_xz_bbox"],
             render_img, render_world_bbox, lidar["xz_polygon"])

    default_transform = None
    for name, fn in METHODS:
        print(f"  [autosnip] Running {name} ...")
        try:
            transform_fn, _, note, reg_debug = fn(*_args)
            # Save all numpy debug images from reg_debug with method name prefix
            for key, val in reg_debug.items():
                if isinstance(val, np.ndarray):
                    dbg_path = os.path.join(output_dir,
                                            f"debug_SU{su_number}_{name}_{key}.png")
                    cv2.imwrite(dbg_path, val)
            print(f"  [{name}] {note}")
            if name == "rgb_pca":
                default_transform = transform_fn
        except Exception as e:
            print(f"  [{name}] failed: {e}")

    if default_transform is None:
        return None, None

    xz_polygons = lidar.get("xz_polygons") or [lidar["xz_polygon"]]
    yellow_worlds = [default_transform(p) for p in xz_polygons]
    return default_transform, yellow_worlds


# ---------------------------------------------------------------------------
# Mode 2: Manual Snip — annotated PLY ortho PNG
# ---------------------------------------------------------------------------

def run_manual_snip(annotated_path: str, render_img: np.ndarray,
                    render_world_bbox: tuple) -> list:
    """
    Extract annotation polygon from a hand-annotated PLY ortho image.
    The annotation should be drawn as black strokes over the ortho image.
    Maps ortho pixel coords → PLY world coords via content-bbox normalisation.
    Returns list of (N,2) float arrays in world (x,y) coords.
    """
    ann = cv2.imread(annotated_path)
    if ann is None:
        raise ValueError(f"Cannot read annotated image: {annotated_path}")

    gray = cv2.cvtColor(ann, cv2.COLOR_BGR2GRAY)

    # Detect black annotation pixels
    black_mask = (gray < 50).astype(np.uint8) * 255
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    # Fill stroke outlines to solid regions
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(ann.shape[:2], np.uint8)
    for c in contours:
        if cv2.contourArea(c) > 2000:
            cv2.fillPoly(filled, [c], 255)

    contours2, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_px = [c.reshape(-1, 2).astype(float)
                for c in contours2 if cv2.contourArea(c) > 5000]
    if not polys_px:
        raise ValueError(f"No annotation polygon found in {annotated_path}")

    # Content bbox of ortho: non-black region spans the full PLY world extent
    ys, xs = np.where(gray > 15)
    if len(xs) == 0:
        raise ValueError(f"Ortho image appears completely black: {annotated_path}")
    ox0, oy0 = int(xs.min()), int(ys.min())
    ox1, oy1 = int(xs.max()), int(ys.max())
    ow, oh = ox1 - ox0, oy1 - oy0

    wx0, wy0, wx1, wy1 = render_world_bbox
    yellow_worlds = []
    for pts in polys_px:
        world = np.empty((len(pts), 2))
        world[:, 0] = wx0 + (pts[:, 0] - ox0) / ow * (wx1 - wx0)
        # Y flipped: image top → world max Y
        world[:, 1] = wy1 - (pts[:, 1] - oy0) / oh * (wy1 - wy0)
        yellow_worlds.append(world)
    return yellow_worlds


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_snip_pipeline(json_filepath: str = "input.json") -> None:
    """
    Main entry point. Reads input JSON and runs autosnip or manual snip per annotation.
    Mode is inferred from file extension: .usdz → autosnip, .png → manual snip.

    Callable from any external Python program:
        import auto_snip_script
        auto_snip_script.run_snip_pipeline("input.json")
    """
    json_id = os.path.splitext(os.path.basename(json_filepath))[0]

    with open(json_filepath, "r") as f:
        job_data = json.load(f)

    print("Starting auto-snip processing...")

    for job in job_data:
        top_job     = job["top"]
        bottom_job  = job["bottom"]
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

        print("  Rendering top-down image from top cloud...")
        render_img, render_world_bbox = render_topdown_image(top_cloud, resolution=0.01)
        output_dir = os.path.join(DATA_DIR, json_id)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "debug_topdown_render.png"), render_img)
        print(f"  Render saved: {os.path.join(output_dir, 'debug_topdown_render.png')}")

        for annotation_path in annotations:
            ext = os.path.splitext(annotation_path)[1].lower()
            print(f"\n  Processing annotation: {annotation_path}")

            # ------------------------------------------------------------------
            # AUTOSNIP: LiDAR USDZ with physically-painted yellow annotation
            # ------------------------------------------------------------------
            if ext == '.usdz':
                try:
                    lidar = auto_snip_lidar.process_usdz(annotation_path)
                except Exception as e:
                    print(f"  Error processing USDZ: {e}")
                    continue

                su_number = lidar["su_name"]
                print(f"  SU name from USDZ: {su_number}")

                _li_disp = lidar.get("lidar_render_display", lidar["lidar_render"])
                cv2.imwrite(os.path.join(output_dir, "debug_lidar_render.png"), _li_disp)
                auto_snip_lidar.save_lidar_debug(
                    _li_disp, lidar["xz_polygon"],
                    lidar["lidar_xz_bbox"],
                    os.path.join(output_dir, f"debug_SU{su_number}_lidar_yellow.png"),
                    xz_polygons=lidar.get("xz_polygons"),
                )

                _, yellow_worlds = run_autosnip(
                    lidar, render_img, render_world_bbox, output_dir, su_number
                )
                if yellow_worlds is None:
                    print(f"  All registration methods failed for SU {su_number}, skipping.")
                    continue

            # ------------------------------------------------------------------
            # MANUAL SNIP: hand-annotated PLY ortho PNG with black stroke outline
            # ------------------------------------------------------------------
            elif ext == '.png':
                su_number = parse_su_number(annotation_path) or job.get("su", "unknown")
                print(f"  SU number: {su_number}")

                try:
                    yellow_worlds = run_manual_snip(
                        annotation_path, render_img, render_world_bbox
                    )
                except Exception as e:
                    print(f"  Error in manual snip: {e}")
                    continue

                # Debug: ortho with extracted polygon overlaid
                ann_dbg = cv2.imread(annotation_path).copy()
                rH, rW = render_img.shape[:2]
                rx0, ry0, rx1, ry1 = render_world_bbox
                for yw in yellow_worlds:
                    rpx = np.empty((len(yw), 2), dtype=int)
                    rpx[:, 0] = np.clip(
                        ((yw[:, 0] - rx0) / (rx1 - rx0) * rW).astype(int), 0, rW - 1)
                    rpx[:, 1] = np.clip(
                        ((ry1 - yw[:, 1]) / (ry1 - ry0) * rH).astype(int), 0, rH - 1)
                    cv2.polylines(ann_dbg, [rpx.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
                dbg_path = os.path.join(output_dir, f"debug_SU{su_number}_manual_annotation.png")
                cv2.imwrite(dbg_path, ann_dbg)
                print(f"  Manual annotation debug saved: {os.path.basename(dbg_path)}")

            else:
                print(f"  Unsupported annotation format '{ext}', skipping.")
                continue

            # ------------------------------------------------------------------
            # Common: print world extent, save snip reference, crop, save
            # ------------------------------------------------------------------
            yellow_world = yellow_worlds[0]
            if len(yellow_worlds) > 1:
                print(f"  Multi-polygon: {len(yellow_worlds)} regions detected")

            print(f"  Polygon world extent: "
                  f"X=[{yellow_world[:,0].min():.3f}, {yellow_world[:,0].max():.3f}]  "
                  f"Y=[{yellow_world[:,1].min():.3f}, {yellow_world[:,1].max():.3f}]")

            rH, rW = render_img.shape[:2]
            rx0, ry0, rx1, ry1 = render_world_bbox

            def _world_to_render_px(yw):
                px = np.empty((len(yw), 2), dtype=int)
                px[:, 0] = np.clip(
                    ((yw[:, 0] - rx0) / (rx1 - rx0) * rW).astype(int), 0, rW - 1)
                px[:, 1] = np.clip(
                    ((ry1 - yw[:, 1]) / (ry1 - ry0) * rH).astype(int), 0, rH - 1)
                return px

            highlight = render_img.copy()
            su_mask = np.zeros((rH, rW), dtype=np.uint8)
            for yw in yellow_worlds:
                rpx = _world_to_render_px(yw)
                cv2.fillPoly(su_mask, [rpx.reshape(-1, 1, 2)], 255)
            highlight[su_mask == 0] = (
                highlight[su_mask == 0].astype(float) * 0.35).astype(np.uint8)
            for yw in yellow_worlds:
                rpx = _world_to_render_px(yw)
                cv2.polylines(highlight, [rpx.reshape(-1, 1, 2)], True, (0, 255, 0), 4)

            snip_ref = np.hstack([render_img, highlight])
            ref_path = os.path.join(output_dir, f"debug_SU{su_number}_snip_reference.png")
            cv2.imwrite(ref_path, snip_ref)
            print(f"  Snip reference saved: {os.path.basename(ref_path)}")

            print("  Cropping top cloud...")
            top_cropped    = crop_cloud_by_polygon_2d(top_cloud, yellow_worlds)
            print("  Cropping bottom cloud...")
            bottom_cropped = crop_cloud_by_polygon_2d(bottom_cloud, yellow_worlds)

            if top_cropped is None or bottom_cropped is None:
                print(f"  Error: crop produced empty cloud for SU {su_number}, skipping.")
                continue

            print(f"  Top cropped: {top_cropped.size()} pts  "
                  f"Bottom cropped: {bottom_cropped.size()} pts")

            top_path = save_cleaned_cloud(
                top_cropped,    output_dir, top_id,    su_number, is_top=False)
            bot_path = save_cleaned_cloud(
                bottom_cropped, output_dir, bottom_id, su_number, is_top=False)
            print(f"  Saved: {os.path.basename(top_path)}")
            print(f"  Saved: {os.path.basename(bot_path)}")

            poly_path = os.path.join(output_dir, f"su_{su_number}_polygon.npy")
            np.save(poly_path, yellow_world)
            print(f"  Saved polygon: {os.path.basename(poly_path)}")

    print("\nAuto-snip complete.")


if __name__ == "__main__":
    run_snip_pipeline(sys.argv[1] if len(sys.argv) > 1 else "input.json")
