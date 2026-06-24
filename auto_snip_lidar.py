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
