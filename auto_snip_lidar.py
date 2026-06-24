"""
LIDAR-based annotation extraction for auto_snip_script.py.

USDZ files (iPhone/iPad LiDAR scans) contain a 3D textured mesh where the
archaeologist physically painted yellow marks on the excavation surface.
These marks are baked into the UV color texture.

Pipeline:
  1. Parse the USDZ mesh (vertices, UV coords, face indices) from the binary
     USDC via `usdcat` → USDA text. Do this once and reuse for both yellow
     detection and top-down rendering.
  2. For each triangle face, compute its UV centroid and sample the color
     texture. Faces whose texture is yellow are the annotation.
  3. Collect 3D centroids of yellow faces → annotation cloud in LiDAR space
     (Y-up, local origin, metres). Project to XZ horizontal plane.
  4. Render the LiDAR mesh top-down (XZ projection) with RGB texture colors.
  5. Register this render to the PLY point cloud top-down render using SIFT.
     Both are orthographic top-down views of the same physical scene, so SIFT
     finds many true correspondences (unlike photo-vs-render).
  6. Apply the SIFT homography to transform the yellow XZ polygon into PLY
     world XY coordinates for cropping.
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
import zipfile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_float_array(usda_text: str, key: str, components: int) -> np.ndarray:
    """
    Parse a flat numeric array from a USDA attribute, e.g.:
        point3f[] points = [(x,y,z), ...]
        texCoord2f[] primvars:st = [(u,v), ...]
    Returns (N, components) float32 array. Fast: uses np.fromstring after
    stripping parentheses, avoiding per-element regex.
    """
    start = usda_text.find(f'{key} = [')
    if start == -1:
        raise ValueError(f"Key not found in USDA: '{key}'")
    start += len(f'{key} = [')
    depth, end = 1, start
    while depth:
        c = usda_text[end]
        if c == '[':   depth += 1
        elif c == ']': depth -= 1
        end += 1
    raw = usda_text[start:end - 1].replace('(', '').replace(')', '')
    arr = np.fromstring(raw, sep=',', dtype=np.float32)
    return arr.reshape(-1, components)


def _parse_int_array(usda_text: str, key: str) -> np.ndarray:
    """Parse a flat int array: int[] faceVertexIndices = [0, 1, 2, ...]"""
    start = usda_text.find(f'{key} = [')
    if start == -1:
        raise ValueError(f"Key not found in USDA: '{key}'")
    start += len(f'{key} = [')
    depth, end = 1, start
    while depth:
        c = usda_text[end]
        if c == '[':   depth += 1
        elif c == ']': depth -= 1
        end += 1
    raw = usda_text[start:end - 1]
    return np.fromstring(raw, sep=',', dtype=np.int32)


def _yellow_mask_bgr(bgr_pixels: np.ndarray) -> np.ndarray:
    """Boolean mask: yellow paint pixels (H=20-40, S>150, V>100) in (N,3) BGR input.
    S>150 (not S>80) is required to exclude warm-toned limestone/soil."""
    hsv = cv2.cvtColor(bgr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return (
        (hsv[:, 0] >= 20) & (hsv[:, 0] <= 40) &
        (hsv[:, 1] > 150) &
        (hsv[:, 2] > 100)
    )


def _yellow_xz_polygon(
    xz_pts: np.ndarray,
    cell_size: float = 0.10,
    simplify_m: float = 0.20,
) -> tuple:
    """
    Rasterise yellow XZ centroids → binary grid → contour of the largest
    connected cluster → simplified polygon in XZ world coords.

    Replaces the old convex-hull approach: the contour follows the actual
    painted boundary including concavities, so the snip region is much tighter.

    Args:
        xz_pts:     (N, 2) XZ centroid coords of yellow faces
        cell_size:  grid resolution in metres (default 10 cm).
                    Smaller = finer contour detail.
        simplify_m: polygon simplification tolerance in metres (default 20 cm).
                    Reduces pixel-grid stairstep noise while preserving shape.

    Returns:
        polygon_xz:   (M, 2) float array — contour polygon in XZ world coords
        filtered_pts: (K, 2) float array — centroids belonging to largest cluster
    """
    if len(xz_pts) == 0:
        return xz_pts, xz_pts

    x0, z0 = float(xz_pts[:, 0].min()), float(xz_pts[:, 1].min())
    col = ((xz_pts[:, 0] - x0) / cell_size).astype(np.int32)
    row = ((xz_pts[:, 1] - z0) / cell_size).astype(np.int32)
    W, H = int(col.max()) + 1, int(row.max()) + 1

    grid = np.zeros((H, W), dtype=np.uint8)
    grid[row, col] = 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    if n_labels < 2:
        # Only background — fall back to convex hull of all points
        hull_idx = cv2.convexHull(xz_pts.astype(np.float32).reshape(-1, 1, 2),
                                  returnPoints=False)
        return xz_pts[hull_idx.flatten()].astype(float), xz_pts

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cluster_mask = (labels == largest).astype(np.uint8) * 255

    keep = labels[row, col] == largest
    filtered_pts = xz_pts[keep]
    print(f"  Cluster filter: {keep.sum()} / {len(xz_pts)} yellow centroids in largest cluster")

    # Extract outer contour of the cluster mask
    contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        hull_idx = cv2.convexHull(filtered_pts.astype(np.float32).reshape(-1, 1, 2),
                                  returnPoints=False)
        return filtered_pts[hull_idx.flatten()].astype(float), filtered_pts

    cnt = max(contours, key=cv2.contourArea)

    # Simplify: epsilon in grid pixels → metres / cell_size
    epsilon_px = simplify_m / cell_size
    cnt_simplified = cv2.approxPolyDP(cnt, epsilon_px, closed=True)

    # Convert pixel (col, row) back to XZ world coords
    poly_col = cnt_simplified[:, 0, 0].astype(float)
    poly_row = cnt_simplified[:, 0, 1].astype(float)
    polygon_xz = np.stack([x0 + poly_col * cell_size,
                           z0 + poly_row * cell_size], axis=1)

    print(f"  Yellow contour: {len(polygon_xz)} vertices "
          f"(cell={cell_size*100:.0f} cm, simplify={simplify_m*100:.0f} cm)")
    return polygon_xz, filtered_pts


def _su_name_from_usdz(zf: zipfile.ZipFile) -> str:
    """Extract SU name from the USDC filename inside the USDZ zip."""
    for name in zf.namelist():
        if name.endswith('.usdc'):
            base = os.path.splitext(os.path.basename(name))[0]
            # e.g. 'SU_22044-22048' -> '22044-22048'
            if base.upper().startswith('SU_'):
                return base[3:]
            return base
    return 'unknown'


# ---------------------------------------------------------------------------
# Main one-pass function
# ---------------------------------------------------------------------------

def process_usdz(usdz_path: str, resolution: float = 0.01) -> dict:
    """
    Parse USDZ once, extract yellow annotation and render top-down image.

    Args:
        usdz_path:  path to the .usdz file
        resolution: metres per pixel for the top-down render

    Returns dict with keys:
        xz_polygon:      (M, 2) float  contour polygon of yellow region in LiDAR XZ space
        xz_pts:          (N, 2) float  all yellow face centroid XZ coords
        lidar_render:    (H, W, 3) uint8 BGR top-down render of the mesh
        lidar_xz_bbox:   (x0, z0, x1, z1) in LiDAR local coords
        lidar_pts:       (V, 3) float32 raw vertex positions (X, Y=up, Z) for DEM computation
        su_name:         str, e.g. '22044-22048', parsed from USDC filename
    """
    print(f"  Opening USDZ: {os.path.basename(usdz_path)}")
    with zipfile.ZipFile(usdz_path) as zf:
        su_name   = _su_name_from_usdz(zf)
        usdc_name = next(n for n in zf.namelist() if n.endswith('.usdc'))
        tex_name  = next(n for n in zf.namelist() if 'color' in n.lower() and n.endswith('.png'))

        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            usdc_path = os.path.join(tmpdir, usdc_name)
            tex_path  = os.path.join(tmpdir, tex_name)

            print("  Converting USDC → USDA ...")
            usda = subprocess.check_output(['usdcat', usdc_path]).decode('utf-8')
            print(f"  USDA size: {len(usda) // 1024} KB")

            print("  Parsing mesh ...")
            pts = _parse_float_array(usda, 'point3f[] points', 3)         # (V, 3) XYZ
            sts = _parse_float_array(usda, 'texCoord2f[] primvars:st', 2) # (V, 2) UV
            fvi = _parse_int_array  (usda, 'int[] faceVertexIndices')     # (3F,)
            n_faces = len(fvi) // 3
            fvi = fvi.reshape(n_faces, 3)
            print(f"  Mesh: {len(pts)} vertices, {n_faces} faces")

            print("  Loading texture ...")
            texture = cv2.imread(tex_path)  # BGR
            if texture is None:
                raise RuntimeError(f"Could not load texture: {tex_path}")
            tex_h, tex_w = texture.shape[:2]
            print(f"  Texture: {tex_w}×{tex_h}")

            # ----------------------------------------------------------------
            # Per-face UV centroid → sample texture color
            # UV: U→x, V→y with V flipped (USD V=0 = bottom, OpenCV row=0 = top)
            # ----------------------------------------------------------------
            uv = sts[fvi].mean(axis=1)                             # (F, 2)
            tex_px = np.clip((uv[:, 0] * tex_w).astype(np.int32), 0, tex_w - 1)
            tex_py = np.clip(((1.0 - uv[:, 1]) * tex_h).astype(np.int32), 0, tex_h - 1)
            face_colors = texture[tex_py, tex_px]                  # (F, 3) BGR

            yellow_mask = _yellow_mask_bgr(face_colors)
            print(f"  Yellow faces: {yellow_mask.sum()} / {n_faces}")

            if not yellow_mask.any():
                raise RuntimeError(
                    "No yellow faces found in USDZ texture. "
                    "Check HSV thresholds (H=20-40, S>150, V>100) or annotation paint color."
                )

            # 3D centroids of yellow faces; project to XZ horizontal plane (Y=up)
            face_centroids = pts[fvi].mean(axis=1)       # (F, 3)
            yellow_3d      = face_centroids[yellow_mask]  # (M, 3)
            xz_pts         = yellow_3d[:, [0, 2]]         # (M, 2) horizontal

            # Build tight contour polygon from yellow centroids.
            # Rasterises to a 10 cm grid, keeps the largest connected cluster,
            # extracts the outer boundary contour, and simplifies to 20 cm tolerance.
            xz_polygon, xz_pts = _yellow_xz_polygon(xz_pts, cell_size=0.10, simplify_m=0.20)
            print(f"  Yellow XZ extent: "
                  f"X=[{xz_pts[:,0].min():.2f}, {xz_pts[:,0].max():.2f}]  "
                  f"Z=[{xz_pts[:,1].min():.2f}, {xz_pts[:,1].max():.2f}]")

            # ----------------------------------------------------------------
            # Top-down render: XZ projection with texture colors.
            # Rasterise full triangles (painter's algorithm, back-to-front
            # in Y) so every pixel inside a face is filled. This eliminates
            # the graininess that vertex-splatting + dilation produces.
            # ----------------------------------------------------------------
            x, z = pts[:, 0], pts[:, 2]
            x0, z0 = float(x.min()), float(z.min())
            x1, z1 = float(x.max()), float(z.max())
            W = int(np.ceil((x1 - x0) / resolution)) + 1
            H = int(np.ceil((z1 - z0) / resolution)) + 1
            print(f"  LIDAR render size: {W}×{H} px at {resolution*100:.0f} cm/px")

            img = np.zeros((H, W, 3), dtype=np.uint8)

            # Project all vertices to image coords
            v_px = np.clip(((x - x0) / resolution).astype(np.int32), 0, W - 1)
            v_pz = np.clip(((z - z0) / resolution).astype(np.int32), 0, H - 1)

            # Per-face: projected vertex coords and centroid texture color
            face_cols = np.stack([v_px[fvi[:, 0]], v_pz[fvi[:, 0]],
                                  v_px[fvi[:, 1]], v_pz[fvi[:, 1]],
                                  v_px[fvi[:, 2]], v_pz[fvi[:, 2]]], axis=1)  # (F, 6)

            # Sort faces back-to-front by average Y (height) so higher
            # surfaces overwrite lower ones (painter's algorithm)
            face_y_avg = pts[fvi, 1].mean(axis=1)
            face_order = np.argsort(face_y_avg)

            print(f"  Rasterising {n_faces} triangles ...")
            for fi in face_order:
                tri = face_cols[fi].reshape(3, 2)
                cv2.fillPoly(img, [tri], face_colors[fi].tolist())

            return {
                "xz_polygon":    xz_polygon,
                "xz_pts":        xz_pts,
                "lidar_render":  img,
                "lidar_xz_bbox": (x0, z0, x1, z1),
                "lidar_pts":     pts,          # (V, 3) float32, Y-up, for DEM registration
                "su_name":       su_name,
            }


# ---------------------------------------------------------------------------
# PCA-based registration: LiDAR render footprint → PLY render footprint
#
# The LiDAR render and PLY render both show the same excavation from above
# at the same 1 cm/px scale, so their non-black pixel footprints have the
# same shape and extent. PCA on each footprint gives the centre, principal
# axis (long axis of the trench), and spread — enough to build a similarity
# transform without needing SIFT feature matches.
#
# This mirrors exactly the strategy used for PNG annotations (PCA on the
# red+blue outline vs. PCA on the PLY render).
# ---------------------------------------------------------------------------

def _pca_footprint(render_bgr: np.ndarray) -> tuple:
    """PCA on non-black pixels of a render. Returns (center, angle_deg, std_main, std_perp)."""
    gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 10)
    pts = np.column_stack([xs, ys]).astype(float)
    if len(pts) < 10:
        raise RuntimeError("PCA: render footprint has fewer than 10 non-black pixels")
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # ascending order
    main  = eigenvectors[:, -1]
    angle = np.degrees(np.arctan2(main[1], main[0]))
    return center, angle, float(np.sqrt(eigenvalues[-1])), float(np.sqrt(eigenvalues[0]))


def register_lidar_to_ply_world(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """
    PCA-based similarity transform: LiDAR render → PLY render → PLY world XY.

    Aligns the non-black footprint of the LiDAR top-down render to the non-black
    footprint of the PLY top-down render using PCA (centre, orientation, scale).
    Both renders show the same physical excavation at 1 cm/px.

    Tries both 180° orientations; picks the one where the yellow polygon lands
    within the PLY render bounds.

    Returns:
        transform_fn: callable (N,2) LiDAR XZ → (N,2) PLY world XY
        debug_img:    BGR image: PLY render with transformed yellow hull in green
        note:         str describing which orientation was chosen
    """
    cx_li, ang_li, std_main_li, std_perp_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, std_main_pl, std_perp_pl = _pca_footprint(ply_render)

    scale = (std_main_pl / std_main_li + std_perp_pl / std_perp_li) / 2
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    print(f"  PCA LiDAR render: center=({cx_li[0]:.0f},{cx_li[1]:.0f}) "
          f"angle={ang_li:.1f}° main={std_main_li:.0f}px perp={std_perp_li:.0f}px")
    print(f"  PCA PLY render:   center=({cx_pl[0]:.0f},{cx_pl[1]:.0f}) "
          f"angle={ang_pl:.1f}° main={std_main_pl:.0f}px perp={std_perp_pl:.0f}px")
    print(f"  PCA scale: {scale:.4f}")

    def _xz_to_lidar_px(xz_pts):
        """LiDAR XZ coords → LiDAR render pixel (col, row)."""
        col = (xz_pts[:, 0] - lx0) / (lx1 - lx0) * lW
        row = (xz_pts[:, 1] - lz0) / (lz1 - lz0) * lH
        return np.stack([col, row], axis=1)

    def _make_M(rot_deg: float) -> np.ndarray:
        """Build 3×3 similarity transform: LiDAR render px → PLY render px."""
        r = np.radians(rot_deg)
        c, s = np.cos(r), np.sin(r)
        tx = cx_pl[0] - scale * (c * cx_li[0] - s * cx_li[1])
        ty = cx_pl[1] - scale * (s * cx_li[0] + c * cx_li[1])
        return np.array([[scale * c, -scale * s, tx],
                         [scale * s,  scale * c, ty],
                         [0, 0, 1]])

    def _apply_M(M, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M @ h.T).T[:, :2]

    def _within_ply_render(rpts, margin=0.15):
        return (rpts[:, 0].min() > -pW * margin and
                rpts[:, 0].max() <  pW * (1 + margin) and
                rpts[:, 1].min() > -pH * margin and
                rpts[:, 1].max() <  pH * (1 + margin))

    # Yellow polygon in LiDAR render pixel space
    yellow_lidar_px = _xz_to_lidar_px(xz_polygon)

    rot_deg = ang_pl - ang_li
    chosen_M, note = None, ""
    for rotation in [rot_deg, rot_deg + 180]:
        M = _make_M(rotation)
        if _within_ply_render(_apply_M(M, yellow_lidar_px)):
            note = f"PCA rotation={rotation:.1f}°"
            chosen_M = M
            break

    if chosen_M is None:
        chosen_M = _make_M(rot_deg)
        note = f"PCA rotation={rot_deg:.1f}° (fallback — polygon outside render bounds)"
    print(f"  {note}")

    def transform_fn(xz_pts: np.ndarray) -> np.ndarray:
        """LiDAR XZ → PLY world XY."""
        lidar_px  = _xz_to_lidar_px(xz_pts)
        ply_px    = _apply_M(chosen_M, lidar_px)
        world     = np.empty_like(ply_px, dtype=float)
        world[:, 0] = px0 + ply_px[:, 0] * (px1 - px0) / pW
        world[:, 1] = py1 - ply_px[:, 1] * (py1 - py0) / pH
        return world

    # Debug: PLY render with transformed yellow hull in green
    ply_px = _apply_M(chosen_M, yellow_lidar_px).astype(np.int32)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px.reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 0), thickness=4)

    return transform_fn, debug_img, note


# ---------------------------------------------------------------------------
# DEM-based registration: height-grid PCA → LiDAR XZ → PLY world XY
#
# RGB-footprint PCA aligns the overall scan outline, which can shift if the
# LiDAR scan and PLY point cloud have different extents (e.g. the LiDAR scan
# includes surrounding area outside the trench). DEM PCA is more stable: it
# aligns the *wall-top* structure — the highest-elevation cells in each scan —
# which corresponds to the same physical edges regardless of scan extent.
#
# Pipeline:
#   1. Bin both point clouds into 5 cm height grids.
#   2. Threshold each grid at the 60th percentile elevation (wall tops).
#   3. PCA on the XZ/XY positions of above-threshold cells.
#   4. Similarity transform from matching PCAs; try 180° ambiguity.
#   5. Optionally save DEM debug images for manual inspection.
# ---------------------------------------------------------------------------

def _compute_lidar_dem_wall_pts(
    lidar_pts: np.ndarray,
    cell_size: float = 0.02,
    pct: float = 30,
    use_floor: bool = True,
    center_frac: float = 1.0,
) -> tuple:
    """
    Build a height grid from LiDAR vertices (Y-up) and return XZ positions
    of floor cells (or wall-top cells).

    pct is a fixed fraction of the normalized elevation RANGE (not a data
    percentile). With use_floor=True (default), selects cells in the BOTTOM
    pct% of the range — the trench floor. With use_floor=False, selects
    the TOP (100-pct)% — the wall/ceiling.

    center_frac restricts selection to the central portion of the DEM grid
    (1.0 = use all cells, default). The LiDAR floor can appear near the scan
    boundary because the scanner is placed inside the trench — the lowest
    elevation area (floor) is wherever the ground was closest to the sensor.

    Returns:
        wall_pts: (K, 2) XZ coords in LiDAR local space
        dem:      (H, W) float32 height grid (NaN where empty)
        origin:   (x0, z0) world origin of grid
    """
    x   = lidar_pts[:, 0].astype(np.float32)
    z   = lidar_pts[:, 2].astype(np.float32)
    elv = lidar_pts[:, 1].astype(np.float32)   # Y = up

    x0, z0 = float(x.min()), float(z.min())
    col = ((x - x0) / cell_size).astype(np.int32)
    row = ((z - z0) / cell_size).astype(np.int32)
    W, H = int(col.max()) + 1, int(row.max()) + 1

    flat = np.full(H * W, -np.inf, dtype=np.float32)
    np.maximum.at(flat, row * W + col, elv)
    dem = flat.reshape(H, W)
    valid = dem > -np.inf
    dem[~valid] = np.nan

    if not valid.any():
        return np.empty((0, 2)), dem, (x0, z0)

    # Restrict to center region; outermost strip is scanner boundary artefacts
    center_mask = np.zeros((H, W), dtype=bool)
    r0 = int(H * (1 - center_frac) / 2)
    r1 = H - r0
    c0 = int(W * (1 - center_frac) / 2)
    c1 = W - c0
    center_mask[r0:r1, c0:c1] = True

    dmin, dmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    dem_norm = (dem - dmin) / max(dmax - dmin, 1e-6)
    thresh_norm = pct / 100.0
    if use_floor:
        wr, wc = np.where(valid & center_mask & (dem_norm <= thresh_norm))
    else:
        wr, wc = np.where(valid & center_mask & (dem_norm >= thresh_norm))
    wall_pts = np.stack([x0 + wc * cell_size, z0 + wr * cell_size], axis=1)
    return wall_pts, dem, (x0, z0)


def _load_geotiff_dem_wall_pts(
    dem_path: str,
    ply_world_bbox: tuple,
    wall_pct: float = 30,
    use_floor: bool = True,
    center_frac: float = 1.0,
) -> tuple:
    """
    Load a GeoTIFF DEM (UTM), extract floor (or wall-top) cells, and return
    their positions converted to PLY local coordinates.

    wall_pct is a fixed fraction of the normalized elevation range. With
    use_floor=True (default), selects cells in the BOTTOM wall_pct% — the
    excavated trench floor. With use_floor=False, selects the TOP (100-wall_pct)%.

    center_frac restricts selection to the central portion of the DEM
    (1.0 = use all cells, default). The excavated pit is not necessarily
    centred in the GeoTIFF — for this site the floor cells are on the right
    edge (col centroid at ~89% of width), so a centre crop removes useful data.

    The DEM is in absolute UTM; the PLY cloud uses a local origin. Both cover
    the same physical area, so the offset is:
        utm_offset = (dem_utm_x_min - ply_x_min, dem_utm_y_min - ply_y_min)

    Args:
        dem_path:       path to GeoTIFF DEM in UTM
        ply_world_bbox: (x0, y0, x1, y1) PLY local bbox from render_topdown_image

    Returns:
        wall_pts: (K, 2) floor/wall-top positions in PLY local XY coords
        dem_img:  uint8 BGR debug image (grayscale DEM + red selected cells)
    """
    from osgeo import gdal
    gdal.UseExceptions()

    ds  = gdal.Open(dem_path)
    gt  = ds.GetGeoTransform()   # (x0_utm, px_w, 0, y0_utm_top, 0, px_h<0)
    W_d = ds.RasterXSize
    H_d = ds.RasterYSize

    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem    = band.ReadAsArray().astype(np.float32)  # (H, W)
    if nodata is not None:
        dem[dem == nodata] = np.nan

    # UTM extent corners
    dem_utm_x0   = gt[0]
    dem_utm_y_top = gt[3]                          # max northing (top row)
    dem_utm_y_bot = gt[3] + gt[5] * H_d           # min northing (bottom row)

    # PLY local extent (same physical area)
    ply_x0, ply_y0, ply_x1, ply_y1 = ply_world_bbox

    # Offset: UTM_val - offset = local_val
    # Both X and Y increase in the same direction (east / north)
    offset_x = dem_utm_x0 - ply_x0
    offset_y = dem_utm_y_bot - ply_y0

    print(f"  GeoTIFF DEM: {W_d}×{H_d} px, cell={gt[1]*100:.3f} cm")
    print(f"    UTM extent: X=[{dem_utm_x0:.2f},{dem_utm_x0+gt[1]*W_d:.2f}] "
          f"Y=[{dem_utm_y_bot:.2f},{dem_utm_y_top:.2f}]")
    print(f"    UTM→local offset: dx={offset_x:.3f}  dy={offset_y:.3f}")

    # Restrict to center region — trench floor is near center; baulks and
    # surrounding terrain dominate the GeoTIFF edges.
    center_mask = np.zeros((H_d, W_d), dtype=bool)
    r0 = int(H_d * (1 - center_frac) / 2)
    r1 = H_d - r0
    c0 = int(W_d * (1 - center_frac) / 2)
    c1 = W_d - c0
    center_mask[r0:r1, c0:c1] = True

    # Normalize to [0,1] and apply fixed fraction threshold so this DEM and
    # the LiDAR DEM both select the same RELATIVE elevation band.
    valid  = ~np.isnan(dem)
    dmin, dmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    dem_norm = np.zeros_like(dem)
    dem_norm[valid] = (dem[valid] - dmin) / max(dmax - dmin, 1e-6)
    thresh_norm = wall_pct / 100.0
    if use_floor:
        wr, wc = np.where(valid & center_mask & (dem_norm <= thresh_norm))
    else:
        wr, wc = np.where(valid & center_mask & (dem_norm >= thresh_norm))

    # Pixel (row, col) → UTM → PLY local
    utm_x   = dem_utm_x0   + wc * gt[1]   # left edge of cell
    utm_y   = dem_utm_y_top + wr * gt[5]  # top edge of cell (gt[5] < 0 → decreasing)
    local_x = utm_x - offset_x
    local_y = utm_y - offset_y

    wall_pts = np.stack([local_x, local_y], axis=1)
    thresh_abs = dmin + thresh_norm * (dmax - dmin)
    direction = "<=" if use_floor else ">="
    label = "floor" if use_floor else "wall-top"
    print(f"    {len(wall_pts)} {label} cells (norm{direction}{thresh_norm:.2f}, ={thresh_abs:.3f} m abs)")
    print(f"    Local extent: X=[{local_x.min():.2f},{local_x.max():.2f}] "
          f"Y=[{local_y.min():.2f},{local_y.max():.2f}]")

    # Debug image: normalized grayscale DEM (0→black, 1→white) + wall-top in red
    img8 = (dem_norm * 255).astype(np.uint8)
    dem_img = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    dem_img[wr, wc] = (0, 0, 255)

    return wall_pts, dem_img


def register_lidar_to_ply_world_dem(
    lidar_pts: np.ndarray,
    lidar_xz_bbox: tuple,
    dem_path: str,
    ply_world_bbox: tuple,
    lidar_render: np.ndarray,
    ply_render: np.ndarray,
    xz_polygon: np.ndarray,
    lidar_cell_size: float = 0.02,
    wall_pct: float = 30,
    use_floor: bool = True,
    lidar_center_frac: float = 1.0,
    ply_center_frac: float = 1.0,
) -> tuple:
    """
    DEM-based PCA similarity transform: LiDAR XZ → PLY world XY.

    Both DEMs are normalized to [0,1] independently, then thresholded at a
    fixed fraction of the elevation range. With use_floor=True (default),
    selects the bottom wall_pct% — the excavated trench floor — which is the
    same physical surface in both datasets. Scale is forced to 1.0.

    lidar_center_frac and ply_center_frac (both default 1.0 = use all cells)
    allow restricting floor-cell selection to the central portion of each DEM.
    For this site the floor cells are near the EDGES of both datasets (LiDAR
    floor is at the scan boundary corner; GeoTIFF floor is at col~89%), so
    centre-cropping is disabled by default — use 1.0 unless you know the floor
    is geometrically central in your specific scan.

    Args:
        lidar_pts:         (V, 3) float32 from process_usdz, Y-up
        lidar_xz_bbox:     (x0, z0, x1, z1) LiDAR local coords
        dem_path:          path to GeoTIFF DEM for the top PLY job (UTM Zone 32N)
        ply_world_bbox:    (x0, y0, x1, y1) PLY local XY extent (from render_topdown_image)
        lidar_render:      BGR top-down render of LiDAR (for debug)
        ply_render:        BGR top-down render of PLY (for debug)
        xz_polygon:        (M, 2) yellow annotation polygon in LiDAR XZ space
        lidar_cell_size:   LiDAR DEM grid resolution in metres (default 2 cm)
        wall_pct:          fixed fraction of normalized range (default 30 = bottom 30%)
        use_floor:         True = select low-elevation floor cells; False = high wall-top cells
        lidar_center_frac: fraction of LiDAR DEM to use (centred), default 1.0 (all)
        ply_center_frac:   fraction of GeoTIFF DEM to use (centred), default 1.0 (all)

    Returns:
        transform_fn: callable (N,2) LiDAR XZ → (N,2) PLY world XY
        debug_img:    BGR image: PLY render with transformed polygon in magenta
        note:         str describing the chosen orientation
        dem_debug:    dict with 'lidar_dem_img' and 'ply_dem_img' uint8 arrays
    """
    # ------------------------------------------------------------------
    # LiDAR DEM from raw vertices (Y-up, XZ horizontal, local metres)
    # ------------------------------------------------------------------
    lidar_wall, lidar_dem, lidar_orig = _compute_lidar_dem_wall_pts(
        lidar_pts, cell_size=lidar_cell_size, pct=wall_pct,
        use_floor=use_floor, center_frac=lidar_center_frac)
    lH, lW = lidar_dem.shape
    label = "floor" if use_floor else "wall-top"
    direction = "<=" if use_floor else ">="
    print(f"  LiDAR DEM: {lW}×{lH} cells at {lidar_cell_size*100:.0f} cm, "
          f"{len(lidar_wall)} {label} cells (norm{direction}{wall_pct/100:.2f}, "
          f"center {lidar_center_frac:.0%})")

    if len(lidar_wall) < 10:
        raise RuntimeError(f"DEM registration: too few LiDAR {label} cells")

    # ------------------------------------------------------------------
    # PLY DEM from GeoTIFF, converted to PLY local XY coords
    # ------------------------------------------------------------------
    ply_wall, ply_dem_img = _load_geotiff_dem_wall_pts(
        dem_path, ply_world_bbox, wall_pct=wall_pct,
        use_floor=use_floor, center_frac=ply_center_frac)

    if len(ply_wall) < 10:
        raise RuntimeError(f"DEM registration: too few PLY {label} cells from GeoTIFF")

    # ------------------------------------------------------------------
    # PCA on wall-top positions (both already in physical metres)
    # ------------------------------------------------------------------
    def _pca2(pts):
        center = pts.mean(axis=0)
        cov    = np.cov(pts.T)
        eigval, eigvec = np.linalg.eigh(cov)
        main   = eigvec[:, -1]
        angle  = float(np.degrees(np.arctan2(main[1], main[0])))
        return center, angle, float(np.sqrt(eigval[-1])), float(np.sqrt(eigval[0]))

    cx_li, ang_li, sml_li, spr_li = _pca2(lidar_wall)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca2(ply_wall)

    # Scale fixed to 1.0: both coordinate systems are in physical metres and
    # represent the same physical scene. PCA spread ratios are unreliable when
    # the two wall-top footprints differ in extent.
    scale = 1.0

    print(f"  DEM PCA LiDAR XZ: center=({cx_li[0]:.3f},{cx_li[1]:.3f}) "
          f"angle={ang_li:.1f}° main={sml_li:.3f}m perp={spr_li:.3f}m")
    print(f"  DEM PCA PLY XY:   center=({cx_pl[0]:.3f},{cx_pl[1]:.3f}) "
          f"angle={ang_pl:.1f}° main={sml_pl:.3f}m perp={spr_pl:.3f}m")
    print(f"  DEM scale: {scale:.4f} (fixed)")

    # ------------------------------------------------------------------
    # Similarity transform: LiDAR (X_l, Z_l) → PLY local (X_p, Y_p)
    # ------------------------------------------------------------------
    def _make_M(rot_deg):
        r = np.radians(rot_deg)
        c, s = np.cos(r), np.sin(r)
        tx = cx_pl[0] - scale * (c * cx_li[0] - s * cx_li[1])
        ty = cx_pl[1] - scale * (s * cx_li[0] + c * cx_li[1])
        return np.array([[scale * c, -scale * s, tx],
                         [scale * s,  scale * c, ty],
                         [0, 0, 1]], dtype=float)

    def _apply(M, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M @ h.T).T[:, :2]

    px0, py0, px1, py1 = ply_world_bbox

    def _within_bounds(world_pts, margin=0.5):
        ext_x = (px1 - px0) * margin
        ext_y = (py1 - py0) * margin
        return (world_pts[:, 0].min() > px0 - ext_x and
                world_pts[:, 0].max() < px1 + ext_x and
                world_pts[:, 1].min() > py0 - ext_y and
                world_pts[:, 1].max() < py1 + ext_y)

    rot_deg = ang_pl - ang_li
    chosen_M, note = None, ""
    for rotation in [rot_deg, rot_deg + 180]:
        M = _make_M(rotation)
        if _within_bounds(_apply(M, xz_polygon)):
            chosen_M = M
            note = f"DEM PCA rotation={rotation:.1f}°"
            break
    if chosen_M is None:
        chosen_M = _make_M(rot_deg)
        note = f"DEM PCA rotation={rot_deg:.1f}° (fallback — polygon outside PLY bounds)"
    print(f"  {note}")

    def transform_fn(xz_pts: np.ndarray) -> np.ndarray:
        return _apply(chosen_M, xz_pts)

    # ------------------------------------------------------------------
    # Debug: PLY render with transformed polygon in magenta
    # ------------------------------------------------------------------
    pH, pW = ply_render.shape[:2]
    world_poly = _apply(chosen_M, xz_polygon)
    ppx = np.clip(((world_poly[:, 0] - px0) / (px1 - px0) * pW).astype(np.int32), 0, pW - 1)
    ppy = np.clip(((py1 - world_poly[:, 1]) / (py1 - py0) * pH).astype(np.int32), 0, pH - 1)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [np.stack([ppx, ppy], axis=1).reshape(-1, 1, 2)],
                  isClosed=True, color=(255, 0, 255), thickness=4)

    # LiDAR DEM debug image — normalized grayscale (0→black, 1→white) to match GeoTIFF debug
    lx0, lz0 = lidar_orig
    valid_l = ~np.isnan(lidar_dem)
    dmin_l, dmax_l = float(np.nanmin(lidar_dem)), float(np.nanmax(lidar_dem))
    dem_norm_l = np.zeros_like(lidar_dem)
    dem_norm_l[valid_l] = (lidar_dem[valid_l] - dmin_l) / max(dmax_l - dmin_l, 1e-6)
    img8_l = (dem_norm_l * 255).astype(np.uint8)
    lidar_dem_img = cv2.cvtColor(img8_l, cv2.COLOR_GRAY2BGR)
    wc_l = np.clip(((lidar_wall[:, 0] - lx0) / lidar_cell_size).astype(np.int32), 0, lW - 1)
    wr_l = np.clip(((lidar_wall[:, 1] - lz0) / lidar_cell_size).astype(np.int32), 0, lH - 1)
    lidar_dem_img[wr_l, wc_l] = (0, 0, 255)

    dem_debug = {
        "lidar_dem_img": lidar_dem_img,
        "ply_dem_img":   ply_dem_img,
    }

    return transform_fn, debug_img, note, dem_debug


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def save_lidar_debug(
    lidar_render: np.ndarray,
    xz_polygon: np.ndarray,
    lidar_xz_bbox: tuple,
    out_path: str,
) -> None:
    """Save LiDAR top-down render with yellow polygon overlaid in green."""
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    lH, lW = lidar_render.shape[:2]
    px = ((xz_polygon[:, 0] - lx0) / (lx1 - lx0) * lW).astype(np.int32)
    pz = ((xz_polygon[:, 1] - lz0) / (lz1 - lz0) * lH).astype(np.int32)
    debug = lidar_render.copy()
    cv2.polylines(debug, [np.stack([px, pz], axis=1).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 0), thickness=4)
    cv2.imwrite(out_path, debug)
    print(f"  Saved: {out_path}")
