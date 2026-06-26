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


def _yellow_xz_polygons(
    xz_pts: np.ndarray,
    cell_size: float = 0.10,
    simplify_m: float = 0.20,
    min_area_frac: float = 0.10,
) -> list:
    """
    Rasterise yellow XZ centroids → binary grid → contours of ALL significant
    connected clusters → simplified polygons in XZ world coords.

    All clusters with area ≥ min_area_frac * largest_cluster_area are kept.
    Returns a list of (polygon_xz, cluster_pts) tuples, sorted largest first.
    """
    if len(xz_pts) == 0:
        return []

    x0, z0 = float(xz_pts[:, 0].min()), float(xz_pts[:, 1].min())
    col = ((xz_pts[:, 0] - x0) / cell_size).astype(np.int32)
    row = ((xz_pts[:, 1] - z0) / cell_size).astype(np.int32)
    W, H = int(col.max()) + 1, int(row.max()) + 1

    grid = np.zeros((H, W), dtype=np.uint8)
    grid[row, col] = 255
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    if n_labels < 2:
        hull_idx = cv2.convexHull(xz_pts.astype(np.float32).reshape(-1, 1, 2),
                                  returnPoints=False)
        return [(xz_pts[hull_idx.flatten()].astype(float), xz_pts)]

    foreground_stats = stats[1:]  # skip background label 0
    max_area = foreground_stats[:, cv2.CC_STAT_AREA].max()
    min_area = max_area * min_area_frac

    # Sort clusters largest-first
    order = np.argsort(foreground_stats[:, cv2.CC_STAT_AREA])[::-1]
    epsilon_px = simplify_m / cell_size

    result = []
    for idx in order:
        label_id = idx + 1  # +1 because we skipped background
        area = foreground_stats[idx, cv2.CC_STAT_AREA]
        if area < min_area:
            break

        cluster_mask = (labels == label_id).astype(np.uint8) * 255
        keep = labels[row, col] == label_id
        cluster_pts = xz_pts[keep]

        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            hull_idx = cv2.convexHull(cluster_pts.astype(np.float32).reshape(-1, 1, 2),
                                      returnPoints=False)
            result.append((cluster_pts[hull_idx.flatten()].astype(float), cluster_pts))
            continue

        cnt = max(contours, key=cv2.contourArea)
        cnt_simplified = cv2.approxPolyDP(cnt, epsilon_px, closed=True)
        poly_col = cnt_simplified[:, 0, 0].astype(float)
        poly_row = cnt_simplified[:, 0, 1].astype(float)
        polygon_xz = np.stack([x0 + poly_col * cell_size,
                               z0 + poly_row * cell_size], axis=1)
        result.append((polygon_xz, cluster_pts))

    print(f"  Yellow clusters: {len(result)} kept "
          f"(≥{min_area_frac*100:.0f}% of largest, cell={cell_size*100:.0f} cm)")
    for i, (poly, pts) in enumerate(result):
        print(f"    Cluster {i+1}: {len(poly)} vertices, {len(pts)} centroids")
    return result


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
        xz_polygon:      (M, 2) float  contour polygon of largest yellow region (for registration)
        xz_polygons:     list of (M, 2) arrays — one per detected yellow cluster (for crop union)
        xz_pts:          (N, 2) float  all yellow face centroid XZ coords (all clusters)
        lidar_render:    (H, W, 3) uint8 BGR top-down render of the mesh
        lidar_xz_bbox:   (x0, z0, x1, z1) in LiDAR local coords
        lidar_pts:       (V, 3) float32 raw vertex positions (X, Y=up, Z) for DEM computation
        su_name:         str, e.g. '22044-22048', parsed from USDC filename
    """
    print(f"  Opening USDZ: {os.path.basename(usdz_path)}")
    with zipfile.ZipFile(usdz_path) as zf:
        su_name   = _su_name_from_usdz(zf)
        usdc_name = next(n for n in zf.namelist() if n.endswith('.usdc'))
        # Pick highest-resolution colour texture (largest file size)
        color_pngs = [n for n in zf.namelist() if 'color' in n.lower() and n.endswith('.png')]
        if not color_pngs:
            raise RuntimeError("No color PNG texture found in USDZ")
        tex_name = max(color_pngs, key=lambda n: zf.getinfo(n).file_size)

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
            # Per-vertex UV → texture color, averaged across face vertices.
            # Sampling at each vertex then averaging gives more accurate color
            # than sampling at the UV centroid (avoids seam/mip bias).
            # UV: U→x, V→y with V flipped (USD V=0 = bottom, OpenCV row=0 = top)
            # ----------------------------------------------------------------
            tex_px_v = np.clip((sts[:, 0] * tex_w).astype(np.int32), 0, tex_w - 1)
            tex_py_v = np.clip(((1.0 - sts[:, 1]) * tex_h).astype(np.int32), 0, tex_h - 1)
            vertex_colors = texture[tex_py_v, tex_px_v]            # (V, 3) BGR
            face_colors = vertex_colors[fvi].mean(axis=1).astype(np.uint8)  # (F, 3) BGR

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

            # Build contour polygons from yellow centroids — all significant clusters.
            clusters = _yellow_xz_polygons(xz_pts, cell_size=0.05, simplify_m=0.15)
            if not clusters:
                raise RuntimeError("No yellow polygon clusters found")
            xz_polygons = [poly for poly, _ in clusters]
            # Combine all cluster pts for bbox/render; largest cluster polygon for registration
            xz_pts      = np.concatenate([pts for _, pts in clusters], axis=0)
            xz_polygon  = xz_polygons[0]  # largest cluster — used for registration PCA
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

            # Fill small inter-triangle gaps without expanding the boundary
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            return {
                "xz_polygon":    xz_polygon,   # largest cluster — for registration
                "xz_polygons":   xz_polygons, # all clusters — for crop union
                "xz_pts":        xz_pts,
                "lidar_render":  img,
                "lidar_xz_bbox": (x0, z0, x1, z1),
                "lidar_pts":     pts,
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


def _draw_pca_axes_on_render(
    img: np.ndarray,
    center,
    angle_deg: float,
    std_main: float,
    std_perp: float,
    label: str = "",
) -> np.ndarray:
    """Return a copy of img with PCA centre (red dot), main axis (yellow), minor axis (magenta)."""
    out = img.copy()
    cx, cy = float(center[0]), float(center[1])
    icx, icy = int(round(cx)), int(round(cy))
    r = np.radians(angle_deg)
    dx,  dy  =  np.cos(r),  np.sin(r)     # main axis direction
    dpx, dpy = -np.sin(r),  np.cos(r)     # minor axis (perpendicular)

    Lmain = int(std_main * 3)
    Lperp = int(std_perp * 3)

    # main axis — yellow
    cv2.line(out, (icx - int(dx * Lmain), icy - int(dy * Lmain)),
                  (icx + int(dx * Lmain), icy + int(dy * Lmain)), (0, 255, 255), 4)
    # minor axis — magenta
    cv2.line(out, (icx - int(dpx * Lperp), icy - int(dpy * Lperp)),
                  (icx + int(dpx * Lperp), icy + int(dpy * Lperp)), (255, 0, 255), 3)
    # centre dot — red with white ring
    cv2.circle(out, (icx, icy), 12, (0, 0, 255), -1)
    cv2.circle(out, (icx, icy), 14, (255, 255, 255), 2)

    # angle label
    cv2.putText(out, f"{angle_deg:.1f} deg", (icx + 18, icy - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(out, f"{angle_deg:.1f} deg", (icx + 18, icy - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    if label:
        cv2.putText(out, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6)
        cv2.putText(out, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    return out


def _make_lidar_vs_result(
    lidar_render: np.ndarray,
    yellow_lidar_px: np.ndarray,
    ply_render: np.ndarray,
    ply_result_px: np.ndarray,
    label: str = "PLY result",
) -> np.ndarray:
    """Side-by-side: LiDAR render (cyan annotation) | PLY render (green annotation)."""
    li = lidar_render.copy()
    cv2.polylines(li, [yellow_lidar_px.astype(np.int32).reshape(-1, 1, 2)],
                  True, (0, 255, 255), 3)
    cv2.putText(li, "LiDAR annotation", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
    cv2.putText(li, "LiDAR annotation", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    pl = ply_render.copy()
    cv2.polylines(pl, [ply_result_px.astype(np.int32).reshape(-1, 1, 2)],
                  True, (0, 255, 0), 4)
    cv2.putText(pl, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
    cv2.putText(pl, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    if li.shape[0] != pl.shape[0]:
        tH = pl.shape[0]
        tW = int(li.shape[1] * tH / li.shape[0])
        li = cv2.resize(li, (tW, tH))
    return np.hstack([li, pl])


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
        pca_debug:    dict of extra debug images:
                        'pca_axes'         — side-by-side panels: LiDAR | PLY, each with PCA
                                             centre dot (red), main axis (yellow), minor (magenta)
                                             and the yellow annotation polygon (cyan on LiDAR)
                        'rotation_overlay' — PLY render (base) with warped LiDAR blended at 50%
                                             opacity, showing rotation quality and any centre shift
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

    # --- debug_img: PLY render with transformed yellow hull in green ---
    ply_px = _apply_M(chosen_M, yellow_lidar_px).astype(np.int32)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px.reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 0), thickness=4)

    # --- pca_debug['pca_axes']: side-by-side panels with PCA axes ---
    li_panel = _draw_pca_axes_on_render(lidar_render, cx_li, ang_li, std_main_li, std_perp_li, "LiDAR")
    cv2.polylines(li_panel, [yellow_lidar_px.astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 255), thickness=3)

    pl_panel = _draw_pca_axes_on_render(ply_render, cx_pl, ang_pl, std_main_pl, std_perp_pl, "PLY")

    # resize LiDAR panel to PLY panel height for hstack
    if li_panel.shape[0] != pl_panel.shape[0]:
        tH = pl_panel.shape[0]
        tW = int(li_panel.shape[1] * tH / li_panel.shape[0])
        li_panel = cv2.resize(li_panel, (tW, tH))
    pca_axes_img = np.hstack([li_panel, pl_panel])

    lidar_vs_result = _make_lidar_vs_result(
        lidar_render, yellow_lidar_px, ply_render, ply_px, "PLY (RGB PCA)")

    pca_debug = {"pca_axes": pca_axes_img, "lidar_vs_result": lidar_vs_result}

    return transform_fn, debug_img, note, pca_debug


def register_lidar_to_ply_world_dem_center(
    lidar_pts: np.ndarray,
    lidar_xz_bbox: tuple,
    dem_path: str,
    ply_world_bbox: tuple,
    lidar_render: np.ndarray,
    ply_render: np.ndarray,
    xz_polygon: np.ndarray,
    pct: float = 30,
) -> tuple:
    """
    Hybrid registration: RGB footprint PCA for rotation + DEM floor centroid for translation.

    The PCA footprint gives a reliable rotation (the scan outline principal axis
    is stable regardless of scan extent) but an unreliable translation (the
    centroid of all non-black pixels drifts with how much extra area each scan
    captured beyond the trench).  The DEM floor centroid — mean XZ/XY position
    of the lowest-elevation cells in each scan — anchors to the physical trench
    floor, which is the same point in both datasets.

    Scale is forced to 1.0 (both datasets are physical metres, same scene).

    Returns (transform_fn, debug_img, note, pca_debug).
    pca_debug keys:
        'pca_axes'        — side-by-side LiDAR|PLY with PCA axes (red dot) and
                            DEM floor centroid (blue cross)
        'lidar_vs_result' — LiDAR annotation | PLY result for visual comparison
    """
    cx_li, ang_li, std_main_li, std_perp_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, std_main_pl, std_perp_pl = _pca_footprint(ply_render)
    scale = 1.0

    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    print(f"  [Hybrid] RGB PCA: LiDAR angle={ang_li:.1f}°  PLY angle={ang_pl:.1f}°  "
          f"rot={ang_pl - ang_li:.1f}°")

    # --- DEM floor centroids (same calls as register_lidar_to_ply_world_dem) ---
    lidar_floor_pts, _, _ = _compute_lidar_dem_wall_pts(lidar_pts, use_floor=True, pct=pct)
    if len(lidar_floor_pts) < 10:
        raise RuntimeError(
            f"Hybrid: too few LiDAR floor cells ({len(lidar_floor_pts)}) — "
            "try raising pct or check DEM quality")

    ply_floor_pts, _ = _load_geotiff_dem_wall_pts(
        dem_path, ply_world_bbox, use_floor=True, wall_pct=pct)
    if len(ply_floor_pts) < 10:
        raise RuntimeError(
            f"Hybrid: too few PLY floor cells ({len(ply_floor_pts)}) — "
            "try raising pct or check GeoTIFF coverage")

    lidar_fc_xz = lidar_floor_pts.mean(axis=0)   # (x, z) in LiDAR XZ local space
    ply_fc_xy   = ply_floor_pts.mean(axis=0)      # (x, y) in PLY world space

    # Convert floor centroids to render pixel space (same mapping used by transform_fn)
    def _xz_to_lidar_px(xz_pts):
        col = (xz_pts[:, 0] - lx0) / (lx1 - lx0) * lW
        row = (xz_pts[:, 1] - lz0) / (lz1 - lz0) * lH
        return np.stack([col, row], axis=1)

    lidar_fc_px = _xz_to_lidar_px(lidar_fc_xz[np.newaxis])[0]   # (col, row)
    ply_fc_px   = np.array([
        (ply_fc_xy[0] - px0) / (px1 - px0) * pW,
        (py1 - ply_fc_xy[1]) / (py1 - py0) * pH,
    ])

    print(f"  [Hybrid] LiDAR floor centroid: XZ=({lidar_fc_xz[0]:.3f},{lidar_fc_xz[1]:.3f})  "
          f"render=({lidar_fc_px[0]:.0f},{lidar_fc_px[1]:.0f})  n={len(lidar_floor_pts)}")
    print(f"  [Hybrid] PLY   floor centroid: XY=({ply_fc_xy[0]:.3f},{ply_fc_xy[1]:.3f})  "
          f"render=({ply_fc_px[0]:.0f},{ply_fc_px[1]:.0f})  n={len(ply_floor_pts)}")

    def _make_M(rot_deg: float) -> np.ndarray:
        r = np.radians(rot_deg)
        c, s = np.cos(r), np.sin(r)
        # Translation anchored to DEM floor centroids, not PCA centroids
        tx = ply_fc_px[0] - scale * (c * lidar_fc_px[0] - s * lidar_fc_px[1])
        ty = ply_fc_px[1] - scale * (s * lidar_fc_px[0] + c * lidar_fc_px[1])
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

    yellow_lidar_px = _xz_to_lidar_px(xz_polygon)
    rot_deg = ang_pl - ang_li
    chosen_M, note = None, ""
    for rotation in [rot_deg, rot_deg + 180]:
        M = _make_M(rotation)
        if _within_ply_render(_apply_M(M, yellow_lidar_px)):
            note = f"Hybrid DEM-centre rotation={rotation:.1f}°"
            chosen_M = M
            break
    if chosen_M is None:
        chosen_M = _make_M(rot_deg)
        note = f"Hybrid DEM-centre rotation={rot_deg:.1f}° (fallback)"
    print(f"  {note}")

    def transform_fn(xz_pts: np.ndarray) -> np.ndarray:
        lidar_px = _xz_to_lidar_px(xz_pts)
        ply_px   = _apply_M(chosen_M, lidar_px)
        world    = np.empty_like(ply_px, dtype=float)
        world[:, 0] = px0 + ply_px[:, 0] * (px1 - px0) / pW
        world[:, 1] = py1 - ply_px[:, 1] * (py1 - py0) / pH
        return world

    # --- debug_img: PLY render with green polygon ---
    ply_px_result = _apply_M(chosen_M, yellow_lidar_px).astype(np.int32)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_result.reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 0), thickness=4)

    # --- pca_axes: PCA axes + DEM floor centroid markers on both panels ---
    li_panel = _draw_pca_axes_on_render(
        lidar_render, cx_li, ang_li, std_main_li, std_perp_li, "LiDAR")
    cv2.polylines(li_panel, [yellow_lidar_px.astype(np.int32).reshape(-1, 1, 2)],
                  True, (0, 255, 255), 3)
    cv2.drawMarker(li_panel, (int(lidar_fc_px[0]), int(lidar_fc_px[1])),
                   (255, 100, 0), cv2.MARKER_CROSS, 40, 4)

    pl_panel = _draw_pca_axes_on_render(
        ply_render, cx_pl, ang_pl, std_main_pl, std_perp_pl, "PLY")
    cv2.drawMarker(pl_panel, (int(ply_fc_px[0]), int(ply_fc_px[1])),
                   (255, 100, 0), cv2.MARKER_CROSS, 40, 4)

    if li_panel.shape[0] != pl_panel.shape[0]:
        tH = pl_panel.shape[0]
        tW = int(li_panel.shape[1] * tH / li_panel.shape[0])
        li_panel = cv2.resize(li_panel, (tW, tH))
    pca_axes_img = np.hstack([li_panel, pl_panel])

    lidar_vs_result = _make_lidar_vs_result(
        lidar_render, yellow_lidar_px, ply_render, ply_px_result,
        "PLY (hybrid: PCA rot + DEM floor centroid)")

    return transform_fn, debug_img, note, {
        "pca_axes": pca_axes_img,
        "lidar_vs_result": lidar_vs_result,
    }


# ---------------------------------------------------------------------------
# Experimental registration methods (branch: automate-snipping-fix-shift)
# All share this return signature:
#   (transform_fn, debug_img, note, {"lidar_vs_result": img})
# ---------------------------------------------------------------------------

def _pca_rotation_candidates(lidar_render, ply_render, scale=1.0):
    """Return [(rot_deg, c, s, tx, ty), ...] for both 180° ambiguity options."""
    cx_li, ang_li, sml_li, spr_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca_footprint(ply_render)
    rot0 = ang_pl - ang_li
    out = []
    for rot in [rot0, rot0 + 180]:
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx = cx_pl[0] - scale*(c*cx_li[0] - s*cx_li[1])
        ty = cx_pl[1] - scale*(s*cx_li[0] + c*cx_li[1])
        out.append((rot, c, s, tx, ty))
    return out, (cx_li, ang_li), (cx_pl, ang_pl)


# Method 1 ─────────────────────────────────────────────────────────────────
def register_lidar_to_ply_world_phase_corr(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """PCA rotation + phase-correlation residual translation on Canny edge images.

    After rotating LiDAR into PLY space via PCA, FFT cross-correlates the two
    Canny edge images to find the remaining translational offset.
    """
    cx_li, ang_li, sml_li, spr_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca_footprint(ply_render)
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    ply_gray  = cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY)
    ply_edges = cv2.Canny(ply_gray, 30, 120).astype(np.float32)
    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    rot0 = ang_pl - ang_li
    best_M, best_note, best_resp = None, "", -1.0

    for rot in [rot0, rot0 + 180]:
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx0 = cx_pl[0] - (c*cx_li[0] - s*cx_li[1])
        ty0 = cx_pl[1] - (s*cx_li[0] + c*cx_li[1])
        M0  = np.array([[c, -s, tx0], [s, c, ty0]], dtype=np.float64)

        warped  = cv2.warpAffine(lidar_render, M0, (pW, pH))
        w_gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        w_edges = cv2.Canny(w_gray, 30, 120).astype(np.float32)

        (dx, dy), resp = cv2.phaseCorrelate(ply_edges, w_edges)
        print(f"  [PhaseCorr] rot={rot:.1f}°  offset=({dx:.1f},{dy:.1f})  resp={resp:.4f}")

        if resp > best_resp:
            best_resp = resp
            best_M    = np.array([[c, -s, tx0+dx], [s, c, ty0+dy],
                                   [0,  0,       1]], dtype=np.float64)
            best_note = f"PhaseCorr rot={rot:.1f}° off=({dx:.0f},{dy:.0f}) resp={resp:.3f}"

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    ply_px_r = _apply(best_M, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        lp = _xz_to_li_px(xz_pts)
        pp = _apply(best_M, lp)
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({best_note})")
    print(f"  [PhaseCorr] {best_note}")
    return transform_fn, debug_img, best_note, {"lidar_vs_result": lvr}


# Method 2 ─────────────────────────────────────────────────────────────────
def register_lidar_to_ply_world_prerot_akaze(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """PCA-rotated LiDAR render + AKAZE feature matching for residual correction.

    Pre-applying PCA rotation aligns descriptors to the same orientation before
    matching, which greatly improves inlier count vs. raw SIFT/AKAZE.
    """
    cx_li, ang_li, sml_li, spr_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca_footprint(ply_render)
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    rot0 = ang_pl - ang_li

    akaze = cv2.AKAZE_create()
    kp2, des2 = akaze.detectAndCompute(ply_render, None)
    if des2 is None or len(kp2) < 10:
        raise RuntimeError("AKAZE: too few features in PLY render")

    best_M3, best_note, best_inliers = None, "", -1

    for rot in [rot0, rot0 + 180]:
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx0 = cx_pl[0] - (c*cx_li[0] - s*cx_li[1])
        ty0 = cx_pl[1] - (s*cx_li[0] + c*cx_li[1])
        M0_2x3 = np.array([[c, -s, tx0], [s, c, ty0]], dtype=np.float64)
        M0_3x3 = np.vstack([M0_2x3, [0,0,1]])

        warped = cv2.warpAffine(lidar_render, M0_2x3, (pW, pH))
        wmask  = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8)

        kp1, des1 = akaze.detectAndCompute(warped, wmask)
        n_kp = len(kp1) if kp1 else 0
        if des1 is None or n_kp < 8:
            print(f"  [PreRotAKAZE] rot={rot:.1f}°  too few kps ({n_kp})")
            continue

        bf   = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw  = bf.knnMatch(des1, des2, k=2)
        good = [m for m,n in raw if m.distance < 0.75*n.distance]
        print(f"  [PreRotAKAZE] rot={rot:.1f}°  kp={n_kp}  good={len(good)}")
        if len(good) < 6:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        A_corr, mask_r = cv2.estimateAffinePartial2D(
            pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=12.0,
            confidence=0.995, maxIters=2000)
        if A_corr is None:
            continue
        inliers = int(mask_r.sum()) if mask_r is not None else 0
        print(f"  [PreRotAKAZE] RANSAC inliers={inliers}")
        if inliers > best_inliers:
            best_inliers = inliers
            A3 = np.vstack([A_corr, [0,0,1]])
            best_M3  = A3 @ M0_3x3
            best_note = f"PreRotAKAZE rot={rot:.1f}° inliers={inliers}"

    if best_M3 is None:
        raise RuntimeError("Pre-rotated AKAZE: no valid transform found")

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    ply_px_r = _apply(best_M3, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        lp = _xz_to_li_px(xz_pts)
        pp = _apply(best_M3, lp)
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({best_note})")
    print(f"  [PreRotAKAZE] {best_note}")
    return transform_fn, debug_img, best_note, {"lidar_vs_result": lvr,
                                                "inliers": best_inliers}


# Method 3 ─────────────────────────────────────────────────────────────────
def register_lidar_to_ply_world_annot_boundary(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """Phase correlate the annotation polygon boundary against PLY Canny edges.

    The E-shaped annotation was painted along physical wall boundaries.  Its
    rotated outline should align with structural edges in the PLY render.
    This uses the annotation geometry as a known probe to find the translation
    directly, without relying on general scene statistics.
    """
    cx_li, ang_li, sml_li, spr_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca_footprint(ply_render)
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    rot0 = ang_pl - ang_li

    li_bnd = np.zeros((lH, lW), dtype=np.uint8)
    cv2.polylines(li_bnd, [yellow_lidar_px.astype(np.int32).reshape(-1,1,2)],
                  True, 255, 5)
    li_bnd = cv2.dilate(li_bnd, np.ones((7,7), np.uint8)).astype(np.float32)

    ply_gray   = cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY)
    ply_edges  = cv2.Canny(ply_gray, 30, 120).astype(np.float32)
    ply_edges_d = cv2.dilate(ply_edges, np.ones((5,5), np.uint8)).astype(np.float32)

    best_M, best_note, best_resp = None, "", -1.0

    for rot in [rot0, rot0 + 180]:
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx0 = cx_pl[0] - (c*cx_li[0] - s*cx_li[1])
        ty0 = cx_pl[1] - (s*cx_li[0] + c*cx_li[1])
        M0 = np.array([[c, -s, tx0], [s, c, ty0]], dtype=np.float64)

        rotated_bnd = cv2.warpAffine(li_bnd, M0, (pW, pH))
        (dx, dy), resp = cv2.phaseCorrelate(ply_edges_d, rotated_bnd)
        print(f"  [AnnotBndry] rot={rot:.1f}°  offset=({dx:.1f},{dy:.1f})  resp={resp:.4f}")

        if resp > best_resp:
            best_resp = resp
            best_M    = np.array([[c, -s, tx0+dx], [s, c, ty0+dy],
                                   [0,  0,       1]], dtype=np.float64)
            best_note = f"AnnotBndry rot={rot:.1f}° off=({dx:.0f},{dy:.0f}) resp={resp:.3f}"

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    ply_px_r = _apply(best_M, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        lp = _xz_to_li_px(xz_pts)
        pp = _apply(best_M, lp)
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({best_note})")
    print(f"  [AnnotBndry] {best_note}")
    return transform_fn, debug_img, best_note, {"lidar_vs_result": lvr}


# Method 4 ─────────────────────────────────────────────────────────────────
def _pca_footprint_dist_weighted(render_bgr: np.ndarray) -> tuple:
    """PCA on non-black pixels weighted by (distance from scan boundary)².

    Interior pixels dominate; peripheral regions that differ between LiDAR and
    PLY captures are suppressed, making the centroid more stable.
    """
    gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    w_img = dist ** 2

    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    w   = w_img[ys, xs].astype(np.float64)
    w  /= w.sum()

    center = (pts * w[:, np.newaxis]).sum(axis=0)
    pts_c  = pts - center
    wcov   = (pts_c * w[:, np.newaxis]).T @ pts_c
    eigval, eigvec = np.linalg.eigh(wcov)
    main   = eigvec[:, -1]
    angle  = float(np.degrees(np.arctan2(main[1], main[0])))
    return center, angle, float(np.sqrt(eigval[-1])), float(np.sqrt(eigval[0]))


def register_lidar_to_ply_world_dist_pca(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """RGB PCA registration using a distance-weighted interior centroid.

    Pixels weighted by squared distance from the scan boundary so the centroid
    reflects the shared interior of the scene, not peripheral scan extensions.
    """
    cx_li, ang_li, sml_li, spr_li = _pca_footprint_dist_weighted(lidar_render)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca_footprint_dist_weighted(ply_render)
    scale = 1.0

    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    print(f"  [DistPCA] LiDAR wt-centre=({cx_li[0]:.0f},{cx_li[1]:.0f}) angle={ang_li:.1f}°")
    print(f"  [DistPCA] PLY   wt-centre=({cx_pl[0]:.0f},{cx_pl[1]:.0f}) angle={ang_pl:.1f}°")

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    def _make_M3(rot):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx = cx_pl[0] - (c*cx_li[0] - s*cx_li[1])
        ty = cx_pl[1] - (s*cx_li[0] + c*cx_li[1])
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=np.float64)

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    def _within(rpts, margin=0.15):
        return (rpts[:,0].min() > -pW*margin and rpts[:,0].max() < pW*(1+margin) and
                rpts[:,1].min() > -pH*margin and rpts[:,1].max() < pH*(1+margin))

    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    rot0 = ang_pl - ang_li
    chosen_M3, note = None, ""
    for rot in [rot0, rot0 + 180]:
        M3 = _make_M3(rot)
        if _within(_apply(M3, yellow_lidar_px)):
            note      = f"DistPCA rot={rot:.1f}°"
            chosen_M3 = M3
            break
    if chosen_M3 is None:
        chosen_M3 = _make_M3(rot0)
        note = f"DistPCA rot={rot0:.1f}° (fallback)"
    print(f"  [DistPCA] {note}")

    ply_px_r = _apply(chosen_M3, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        lp = _xz_to_li_px(xz_pts)
        pp = _apply(chosen_M3, lp)
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({note})")
    return transform_fn, debug_img, note, {"lidar_vs_result": lvr}


# ---------------------------------------------------------------------------
# Shared coarse-to-fine pose search (used by chamfer / MI / DEM-ridge methods)
# ---------------------------------------------------------------------------

def _pts_pca_2d(pts: np.ndarray) -> tuple:
    """Return (center(2,), angle_deg) of the principal axis of a 2D point set."""
    center = pts.mean(axis=0)
    pc = pts - center
    cov = pc.T @ pc
    eigval, eigvec = np.linalg.eigh(cov)
    main = eigvec[:, -1]
    angle = float(np.degrees(np.arctan2(main[1], main[0])))
    return center, angle


def _search_pose(cost_fn, src_center, dst_center, rot0,
                 both_flips=True,
                 coarse_rot=(-20, 20, 4), coarse_trans=(-250, 250, 25),
                 fine_rot=(-4, 4, 1), fine_trans=(-25, 25, 5)):
    """Coarse-to-fine search for the rigid pose (rotation + translation) that
    minimises cost_fn(M3).

    The 3x3 matrix maps src px → dst px as a rotation about src_center that lands
    src_center on dst_center + (dx, dy):
        M3 = R, t = dst_center + (dx,dy) - R @ src_center
    cost_fn must return a scalar (lower = better).  Returns (best_M3, best_cost,
    best_rot_deg).
    """
    sx, sy = float(src_center[0]), float(src_center[1])
    dx0, dy0 = float(dst_center[0]), float(dst_center[1])

    def make_M3(rot, dx, dy):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx = dx0 + dx - (c*sx - s*sy)
        ty = dy0 + dy - (s*sx + c*sy)
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=np.float64)

    rot_bases = [rot0, rot0 + 180] if both_flips else [rot0]

    best = (np.inf, None, rot0, 0.0, 0.0)
    cr = np.arange(coarse_rot[0], coarse_rot[1] + 1e-6, coarse_rot[2])
    ct = np.arange(coarse_trans[0], coarse_trans[1] + 1e-6, coarse_trans[2])
    for rb in rot_bases:
        for drot in cr:
            rot = rb + drot
            for dx in ct:
                for dy in ct:
                    M = make_M3(rot, dx, dy)
                    c = cost_fn(M)
                    if c < best[0]:
                        best = (c, M, rot, dx, dy)

    _, _, brot, bdx, bdy = best
    fr = np.arange(fine_rot[0], fine_rot[1] + 1e-6, fine_rot[2])
    ft = np.arange(fine_trans[0], fine_trans[1] + 1e-6, fine_trans[2])
    for drot in fr:
        rot = brot + drot
        for dx in np.arange(bdx + ft[0], bdx + ft[-1] + 1e-6, fine_trans[2]):
            for dy in np.arange(bdy + ft[0], bdy + ft[-1] + 1e-6, fine_trans[2]):
                M = make_M3(rot, dx, dy)
                c = cost_fn(M)
                if c < best[0]:
                    best = (c, M, rot, dx, dy)

    return best[1], best[0], best[2]


def _densify_polygon(pts: np.ndarray, step: float = 3.0) -> np.ndarray:
    """Resample a closed polygon into points spaced ~`step` px along its edges."""
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[i]; b = pts[(i + 1) % n]
        seg = b - a
        L = float(np.hypot(seg[0], seg[1]))
        k = max(int(L / step), 1)
        for j in range(k):
            out.append(a + seg * (j / k))
    return np.asarray(out, dtype=np.float64)


# Method 5 ─────────────────────────────────────────────────────────────────
def register_lidar_to_ply_world_pca_chamfer(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """Chamfer-match the annotation E-shape against PLY wall edges.

    Rotation is fixed to the PCA-validated value (which is known to be correct).
    Only translation is searched — the 2-DOF problem is well-posed even against
    dense edges, while free-rotation search in clutter will snap to any coincident
    edge pattern regardless of semantic correctness.
    """
    cx_li, ang_li, std_main_li, std_perp_li = _pca_footprint(lidar_render)
    cx_pl, ang_pl, std_main_pl, std_perp_pl = _pca_footprint(ply_render)
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    # PLY edge distance transform (distance to nearest wall edge)
    ply_gray  = cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY)
    ply_edges = cv2.Canny(ply_gray, 30, 120)
    ply_edges = cv2.dilate(ply_edges, np.ones((3,3), np.uint8))
    dt = cv2.distanceTransform(255 - ply_edges, cv2.DIST_L2, 5).astype(np.float32)
    PENALTY = float(dt.max()) * 2.0

    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    poly_pts = _densify_polygon(yellow_lidar_px, step=3.0)
    homog = np.column_stack([poly_pts, np.ones(len(poly_pts))])

    def cost_fn(M3):
        pp = (M3 @ homog.T).T[:, :2]
        xs = np.round(pp[:,0]).astype(np.int32)
        ys = np.round(pp[:,1]).astype(np.int32)
        inb = (xs >= 0) & (xs < pW) & (ys >= 0) & (ys < pH)
        if inb.sum() < 0.3 * len(pp):
            return PENALTY
        vals = np.full(len(pp), PENALTY, dtype=np.float32)
        vals[inb] = dt[ys[inb], xs[inb]]
        return float(vals.mean())

    def _within(pts, margin=0.15):
        return (pts[:,0].min() > -pW*margin and pts[:,0].max() < pW*(1+margin) and
                pts[:,1].min() > -pH*margin and pts[:,1].max() < pH*(1+margin))

    def _make_M3_zero_dt(rot):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx = cx_pl[0] - (c*cx_li[0] - s*cx_li[1])
        ty = cx_pl[1] - (s*cx_li[0] + c*cx_li[1])
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=np.float64)

    def _apply0(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    # Determine the correct 180° flip from PCA (polygon must land inside render)
    rot0 = ang_pl - ang_li
    fixed_rot = rot0
    for rot_candidate in [rot0, rot0 + 180]:
        if _within(_apply0(_make_M3_zero_dt(rot_candidate), yellow_lidar_px)):
            fixed_rot = rot_candidate
            break
    print(f"  [Chamfer] locked rot={fixed_rot:.1f}° — searching translation only")

    # Translation-only search: coarse_rot=(0,0,1) → only drot=0 is tried
    best_M3, best_cost, best_rot = _search_pose(
        cost_fn, cx_li, cx_pl, fixed_rot,
        both_flips=False,
        coarse_rot=(0, 0, 1), fine_rot=(0, 0, 1),
        coarse_trans=(-300, 300, 20), fine_trans=(-20, 20, 4))
    note = f"Chamfer rot={best_rot:.1f}° meanDist={best_cost:.2f}px"
    print(f"  [Chamfer] {note}")

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    ply_px_r = _apply(best_M3, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        pp = _apply(best_M3, _xz_to_li_px(xz_pts))
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({note})")

    # PCA axes side-by-side: LiDAR panel (cyan annotation) | PLY panel (green result)
    li_panel = _draw_pca_axes_on_render(
        lidar_render, cx_li, ang_li, std_main_li, std_perp_li, "LiDAR")
    cv2.polylines(li_panel, [yellow_lidar_px.astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 255), thickness=3)
    pl_panel = _draw_pca_axes_on_render(
        ply_render, cx_pl, ang_pl, std_main_pl, std_perp_pl, "PLY")
    cv2.polylines(pl_panel, [ply_px_r.reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 0), thickness=3)
    if li_panel.shape[0] != pl_panel.shape[0]:
        tH = pl_panel.shape[0]
        tW = int(li_panel.shape[1] * tH / li_panel.shape[0])
        li_panel = cv2.resize(li_panel, (tW, tH))
    pca_axes_img = np.hstack([li_panel, pl_panel])

    return transform_fn, debug_img, note, {"lidar_vs_result": lvr, "pca_axes": pca_axes_img}


# ---------------------------------------------------------------------------
# Claude Vision registration — semantic localization + Chamfer precision
#
# Sends a side-by-side composite (LiDAR annotation | PLY render) to the
# Anthropic API.  Claude identifies approximately where the annotated room
# sits in the PLY render and returns its pixel center.  That center seeds a
# narrow Chamfer translation search (±120 px) instead of the PCA centroid,
# which is unreliable on large multi-room sites.
#
# Requires: ANTHROPIC_API_KEY env var, `pip install anthropic`
# ---------------------------------------------------------------------------

def _call_claude_for_center(
    composite_bgr: np.ndarray,
    left_w: int,
    right_w: int,
    right_h: int,
    model: str,
) -> tuple:
    """Call Claude Vision API and return (cx, cy) in the right panel (resized coords)."""
    import anthropic, base64, json as _json

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    ok, buf = cv2.imencode(".png", composite_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    img_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    prompt = (
        "You are helping register two images of the same archaeological excavation.\n\n"
        "The image you see has TWO panels side by side:\n"
        f"  LEFT panel (roughly {left_w}px wide): iPhone LiDAR scan from INSIDE "
        "the trench. The orange/yellow shaded polygon is the annotated sub-unit.\n"
        f"  RIGHT panel ({right_w}×{right_h}px): Top-down photogrammetry of the "
        "SAME site from above. No annotation is shown.\n\n"
        "The two panels show the same physical walls/rooms from different viewpoints:\n"
        "the LEFT sees wall faces from inside; the RIGHT sees wall tops from above.\n\n"
        "Task: Find the approximate pixel center of the region in the RIGHT panel "
        "that corresponds to the yellow/orange annotated area in the LEFT panel.\n"
        "Think about the room layout — which room or area in the top-down view "
        "matches the annotated space in the scan?\n\n"
        "Return ONLY valid JSON, no other text:\n"
        f'{{\"cx\": <int 0..{right_w-1}>, \"cy\": <int 0..{right_h-1}>}}'
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = resp.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:]).rstrip("`").strip()

    try:
        data = _json.loads(raw)
        cx = int(data["cx"])
        cy = int(data["cy"])
        return cx, cy
    except Exception as e:
        raise RuntimeError(f"Claude response parse failed: {e}  raw={raw[:200]!r}")


def register_lidar_to_ply_world_claude_vision(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple:
    """Register LiDAR → PLY using Claude Vision for semantic localization.

    Claude identifies which room/area in the PLY corresponds to the annotated
    region in the LiDAR scan.  Its rough pixel-center estimate seeds a narrow
    Chamfer translation search (±120 px), combining semantic understanding with
    geometric precision.
    """
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:, 0] - lx0) / (lx1 - lx0) * lW,
                         (xz[:, 1] - lz0) / (lz1 - lz0) * lH], axis=1)

    yellow_lidar_px = _xz_to_li_px(xz_polygon)

    # ------------------------------------------------------------------
    # Build composite prompt image — both panels at max 768px long edge
    # ------------------------------------------------------------------
    max_dim = 768
    l_scale = min(max_dim / lW, max_dim / lH, 1.0)
    p_scale = min(max_dim / pW, max_dim / pH, 1.0)
    lsW, lsH = int(lW * l_scale), int(lH * l_scale)
    psW, psH = int(pW * p_scale), int(pH * p_scale)
    l_small = cv2.resize(lidar_render, (lsW, lsH))
    p_small = cv2.resize(ply_render,   (psW, psH))

    # Draw yellow semi-transparent annotation fill on LiDAR panel
    yellow_li_small = (yellow_lidar_px * l_scale).astype(np.int32)
    overlay = l_small.copy()
    cv2.fillPoly(overlay, [yellow_li_small], (0, 200, 255))
    cv2.addWeighted(overlay, 0.4, l_small, 0.6, 0, l_small)
    cv2.polylines(l_small, [yellow_li_small.reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 255), thickness=3)

    # Pad to same height, add divider
    comp_h = max(lsH, psH)
    def _pad(img, h):
        if img.shape[0] >= h:
            return img
        return np.vstack([img, np.zeros((h - img.shape[0], img.shape[1], 3), np.uint8)])

    divider = np.full((comp_h, 6, 3), 80, dtype=np.uint8)
    composite = np.hstack([_pad(l_small, comp_h), divider, _pad(p_small, comp_h)])

    # ------------------------------------------------------------------
    # Ask Claude for the center of the corresponding region in PLY panel
    # ------------------------------------------------------------------
    print(f"  [ClaudeVision] Calling {model} ...")
    cx_small, cy_small = _call_claude_for_center(composite, lsW, psW, psH, model)
    # Scale back from resized PLY coords to original PLY px coords
    cx_ply = cx_small / p_scale
    cy_ply = cy_small / p_scale
    print(f"  [ClaudeVision] Claude center in PLY: ({cx_ply:.0f}, {cy_ply:.0f})")

    # ------------------------------------------------------------------
    # Chamfer refinement seeded at Claude's center (±120 px search)
    # ------------------------------------------------------------------
    cx_li_center = float(yellow_lidar_px[:, 0].mean())
    cy_li_center = float(yellow_lidar_px[:, 1].mean())

    # PCA rotation (locked, same as pca_chamfer)
    cx_li_pca, ang_li, _, _ = _pca_footprint(lidar_render)
    cx_pl_pca, ang_pl, _, _ = _pca_footprint(ply_render)
    rot0 = ang_pl - ang_li

    # Pick 180° flip that keeps polygon inside PLY bounds
    def _make_M3(rot, src_c, dst_c):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx = dst_c[0] - (c * src_c[0] - s * src_c[1])
        ty = dst_c[1] - (s * src_c[0] + c * src_c[1])
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=np.float64)

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    src_li = np.array([cx_li_center, cy_li_center])
    dst_claude = np.array([cx_ply, cy_ply])

    fixed_rot = rot0
    for rot_candidate in [rot0, rot0 + 180]:
        M_test = _make_M3(rot_candidate, src_li, dst_claude)
        pts_test = _apply(M_test, yellow_lidar_px)
        margin = 0.2
        if (pts_test[:, 0].min() > -pW * margin and pts_test[:, 0].max() < pW * (1 + margin) and
                pts_test[:, 1].min() > -pH * margin and pts_test[:, 1].max() < pH * (1 + margin)):
            fixed_rot = rot_candidate
            break

    # Build edge distance transform
    ply_gray  = cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY)
    ply_edges = cv2.Canny(ply_gray, 30, 120)
    ply_edges = cv2.dilate(ply_edges, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(255 - ply_edges, cv2.DIST_L2, 5).astype(np.float32)
    PENALTY = float(dt.max()) * 2.0

    poly_pts = _densify_polygon(yellow_lidar_px, step=3.0)
    homog = np.column_stack([poly_pts, np.ones(len(poly_pts))])

    def cost_fn(M3):
        pp = (M3 @ homog.T).T[:, :2]
        xs = np.round(pp[:, 0]).astype(np.int32)
        ys = np.round(pp[:, 1]).astype(np.int32)
        inb = (xs >= 0) & (xs < pW) & (ys >= 0) & (ys < pH)
        if inb.sum() < 0.3 * len(pp):
            return PENALTY
        vals = np.full(len(pp), PENALTY, dtype=np.float32)
        vals[inb] = dt[ys[inb], xs[inb]]
        return float(vals.mean())

    best_M3, best_cost, best_rot = _search_pose(
        cost_fn, src_li, dst_claude, fixed_rot,
        both_flips=False,
        coarse_rot=(0, 0, 1), fine_rot=(0, 0, 1),
        coarse_trans=(-120, 120, 15), fine_trans=(-15, 15, 3),
    )
    note = f"ClaudeVision+Chamfer rot={best_rot:.1f}° meanDist={best_cost:.2f}px"
    print(f"  [ClaudeVision] {note}")

    # ------------------------------------------------------------------
    # Build transform_fn and debug images
    # ------------------------------------------------------------------
    ply_px_r = _apply(best_M3, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        pp = _apply(best_M3, _xz_to_li_px(xz_pts))
        world = np.empty((len(pp), 2))
        world[:, 0] = px0 + pp[:, 0] * (px1 - px0) / pW
        world[:, 1] = py1 - pp[:, 1] * (py1 - py0) / pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1, 1, 2)], True, (0, 255, 0), 4)

    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({note})")

    return transform_fn, debug_img, note, {
        "lidar_vs_result": lvr,
        "claude_prompt":   composite,
    }


# Method 6 ─────────────────────────────────────────────────────────────────
def _mutual_information(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                        bins: int = 32) -> float:
    """Mutual information between two gray images over a boolean mask."""
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    hist, _, _ = np.histogram2d(av, bv, bins=bins, range=[[0, 256], [0, 256]])
    pab = hist / max(hist.sum(), 1.0)
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)
    nz = pab > 0
    denom = (pa[:, None] * pb[None, :])
    return float((pab[nz] * np.log(pab[nz] / denom[nz])).sum())


def register_lidar_to_ply_world_mutual_info(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
) -> tuple:
    """Maximise mutual information between the PLY and warped LiDAR gray images.

    The textbook multimodal-registration approach: MI rewards statistical
    coupling between the two intensity patterns without requiring them to look
    alike.  Search runs in a downsampled space for speed, then the matrix is
    scaled back to full render resolution.
    """
    cx_li, ang_li, _, _ = _pca_footprint(lidar_render)
    cx_pl, ang_pl, _, _ = _pca_footprint(ply_render)
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    # Downsample to ~400 px max dim for speed
    f = 400.0 / max(pH, pW, lH, lW)
    f = min(f, 1.0)
    li_small = cv2.resize(cv2.cvtColor(lidar_render, cv2.COLOR_BGR2GRAY),
                          None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    pl_small = cv2.resize(cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY),
                          None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    pH_s, pW_s = pl_small.shape
    pl_mask = pl_small > 10

    def cost_fn(M3_small):
        M2 = M3_small[:2, :]
        warped = cv2.warpAffine(li_small, M2, (pW_s, pH_s))
        mask = pl_mask & (warped > 10)
        if mask.sum() < 0.05 * pl_mask.sum():
            return 1e6
        return -_mutual_information(pl_small, warped, mask)

    rot0 = ang_pl - ang_li
    # search in small-image coords: scale the PCA centers down
    src_c = cx_li * f
    dst_c = cx_pl * f
    best_M3s, best_cost, best_rot = _search_pose(
        cost_fn, src_c, dst_c, rot0,
        coarse_trans=(-100, 100, 12), fine_trans=(-12, 12, 3))

    # scale matrix back up to full render px:  M_full = D(1/f) M_small D(f)
    Sf  = np.diag([f, f, 1.0])
    Sif = np.diag([1.0/f, 1.0/f, 1.0])
    best_M3 = Sif @ best_M3s @ Sf
    note = f"MutualInfo rot={best_rot:.1f}° MI={-best_cost:.3f}"
    print(f"  [MutualInfo] {note}")

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    ply_px_r = _apply(best_M3, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        pp = _apply(best_M3, _xz_to_li_px(xz_pts))
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({note})")
    return transform_fn, debug_img, note, {"lidar_vs_result": lvr}


# Method 7 ─────────────────────────────────────────────────────────────────
def _ridge_mask_from_grid(grid: np.ndarray, pct: float = 90.0) -> np.ndarray:
    """High-slope ridge cells of a height grid (NaN = empty).  Returns (rr, cc)."""
    valid = ~np.isnan(grid)
    fill = float(np.nanmin(grid)) if valid.any() else 0.0
    g = np.where(valid, grid, fill).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # suppress artificial gradients at the valid/empty boundary
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
    mag[~inner] = 0.0
    if not inner.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    thr = np.percentile(mag[inner], pct)
    return np.where((mag >= thr) & inner)


def register_lidar_to_ply_world_dem_ridge(
    lidar_pts: np.ndarray,
    lidar_xz_bbox: tuple,
    dem_path: str,
    ply_world_bbox: tuple,
    lidar_render: np.ndarray,
    ply_render: np.ndarray,
    xz_polygon: np.ndarray,
    lidar_cell_size: float = 0.02,
    ridge_pct: float = 90.0,
) -> tuple:
    """Register via DEM slope-ridge structure (walls = high-gradient ridges).

    Wall edges appear as sharp slope ridges in BOTH the LiDAR-derived height grid
    and the GeoTIFF DEM, regardless of absolute elevation or scan extent.  Build
    ridge images for each (in PLY render px), then chamfer-search the LiDAR ridge
    points against the PLY ridge distance transform.
    """
    from osgeo import gdal
    gdal.UseExceptions()

    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    def _xz_to_li_px(xz):
        return np.stack([(xz[:,0]-lx0)/(lx1-lx0)*lW,
                         (xz[:,1]-lz0)/(lz1-lz0)*lH], axis=1)

    # --- LiDAR ridge cells → LiDAR render px ---
    _, lidar_dem, (lgx0, lgz0) = _compute_lidar_dem_wall_pts(
        lidar_pts, cell_size=lidar_cell_size)
    lrr, lcc = _ridge_mask_from_grid(lidar_dem, pct=ridge_pct)
    if len(lrr) == 0:
        raise RuntimeError("DEM-ridge: no LiDAR ridge cells")
    lidar_xz = np.stack([lgx0 + lcc * lidar_cell_size,
                         lgz0 + lrr * lidar_cell_size], axis=1)
    ridge_li_px = _xz_to_li_px(lidar_xz)

    # --- GeoTIFF ridge cells → PLY local XY → PLY render px ---
    ds = gdal.Open(dem_path)
    gt = ds.GetGeoTransform()
    W_d, H_d = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem = band.ReadAsArray().astype(np.float32)
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem_utm_x0 = gt[0]
    dem_utm_y_top = gt[3]
    dem_utm_y_bot = gt[3] + gt[5] * H_d
    offset_x = dem_utm_x0 - px0
    offset_y = dem_utm_y_bot - py0

    grr, gcc = _ridge_mask_from_grid(dem, pct=ridge_pct)
    if len(grr) == 0:
        raise RuntimeError("DEM-ridge: no GeoTIFF ridge cells")
    utm_x = dem_utm_x0 + gcc * gt[1]
    utm_y = dem_utm_y_top + grr * gt[5]
    local_x = utm_x - offset_x
    local_y = utm_y - offset_y
    ply_ridge_px = np.stack([(local_x - px0)/(px1-px0)*pW,
                             (py1 - local_y)/(py1-py0)*pH], axis=1)
    print(f"  [DEMRidge] LiDAR ridge cells={len(lrr)}  GeoTIFF ridge cells={len(grr)}")

    # PLY ridge image → distance transform
    ridge_img = np.zeros((pH, pW), dtype=np.uint8)
    inb = ((ply_ridge_px[:,0] >= 0) & (ply_ridge_px[:,0] < pW) &
           (ply_ridge_px[:,1] >= 0) & (ply_ridge_px[:,1] < pH))
    pr = ply_ridge_px[inb].astype(np.int32)
    ridge_img[pr[:,1], pr[:,0]] = 255
    ridge_img = cv2.dilate(ridge_img, np.ones((3,3), np.uint8))
    dt = cv2.distanceTransform(255 - ridge_img, cv2.DIST_L2, 5).astype(np.float32)
    PENALTY = float(dt.max()) * 2.0

    # subsample LiDAR ridge points for cost speed
    if len(ridge_li_px) > 800:
        idx = np.random.RandomState(0).choice(len(ridge_li_px), 800, replace=False)
        cost_pts = ridge_li_px[idx]
    else:
        cost_pts = ridge_li_px
    homog = np.column_stack([cost_pts, np.ones(len(cost_pts))])

    def cost_fn(M3):
        pp = (M3 @ homog.T).T[:, :2]
        xs = np.round(pp[:,0]).astype(np.int32)
        ys = np.round(pp[:,1]).astype(np.int32)
        ok = (xs >= 0) & (xs < pW) & (ys >= 0) & (ys < pH)
        if ok.sum() < 0.3 * len(pp):
            return PENALTY
        vals = np.full(len(pp), PENALTY, dtype=np.float32)
        vals[ok] = dt[ys[ok], xs[ok]]
        return float(vals.mean())

    # Lock rotation to PCA-validated render-based estimate; search translation only.
    # The DEM ridge PCA is under-constrained (381k GeoTIFF cells vs 239 LiDAR cells),
    # so we don't trust it for rotation — use the render PCA which is known-good.
    cx_li_r, ang_li_r, _, _ = _pca_footprint(lidar_render)
    cx_pl_r, ang_pl_r, _, _ = _pca_footprint(ply_render)
    rot0 = ang_pl_r - ang_li_r

    def _within_r(pts, margin=0.15):
        return (pts[:,0].min() > -pW*margin and pts[:,0].max() < pW*(1+margin) and
                pts[:,1].min() > -pH*margin and pts[:,1].max() < pH*(1+margin))

    def _make_M3_r(rot):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        tx = cx_pl_r[0] - (c*cx_li_r[0] - s*cx_li_r[1])
        ty = cx_pl_r[1] - (s*cx_li_r[0] + c*cx_li_r[1])
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=np.float64)

    def _apply_r(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    yellow_lidar_px_r = _xz_to_li_px(xz_polygon)
    fixed_rot = rot0
    for rc in [rot0, rot0 + 180]:
        if _within_r(_apply_r(_make_M3_r(rc), yellow_lidar_px_r)):
            fixed_rot = rc
            break
    print(f"  [DEMRidge] locked rot={fixed_rot:.1f}° — searching translation only")

    best_M3, best_cost, best_rot = _search_pose(
        cost_fn, cx_li_r, cx_pl_r, fixed_rot,
        both_flips=False,
        coarse_rot=(0, 0, 1), fine_rot=(0, 0, 1),
        coarse_trans=(-300, 300, 20), fine_trans=(-20, 20, 4))
    note = f"DEMRidge rot={best_rot:.1f}° meanDist={best_cost:.2f}px"
    print(f"  [DEMRidge] {note}")

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    yellow_lidar_px = _xz_to_li_px(xz_polygon)
    ply_px_r = _apply(best_M3, yellow_lidar_px).astype(np.int32)

    def transform_fn(xz_pts):
        pp = _apply(best_M3, _xz_to_li_px(xz_pts))
        world = np.empty((len(pp), 2))
        world[:,0] = px0 + pp[:,0]*(px1-px0)/pW
        world[:,1] = py1 - pp[:,1]*(py1-py0)/pH
        return world

    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [ply_px_r.reshape(-1,1,2)], True, (0,255,0), 4)
    lvr = _make_lidar_vs_result(lidar_render, yellow_lidar_px, ply_render, ply_px_r,
                                f"PLY ({note})")
    return transform_fn, debug_img, note, {"lidar_vs_result": lvr}


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

    # Normalise elevation relative to the center region only, so pct selects
    # the bottom fraction of whatever elevation range exists in the center
    # (not the global range, which may be dominated by cells outside the center).
    center_valid = valid & center_mask
    if center_valid.any():
        dmin = float(np.nanmin(dem[center_valid]))
        dmax = float(np.nanmax(dem[center_valid]))
    else:
        dmin, dmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    dem_norm = np.full_like(dem, np.nan)
    dem_norm[center_valid] = (dem[center_valid] - dmin) / max(dmax - dmin, 1e-6)
    thresh_norm = pct / 100.0
    if use_floor:
        wr, wc = np.where(center_valid & (dem_norm <= thresh_norm))
    else:
        wr, wc = np.where(center_valid & (dem_norm >= thresh_norm))
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

    # Normalise elevation relative to the center region only, so pct selects
    # the bottom fraction of whatever elevation range exists in the center
    # (not the global range, which is dominated by surrounding terrain).
    valid  = ~np.isnan(dem)
    center_valid = valid & center_mask
    if center_valid.any():
        dmin = float(np.nanmin(dem[center_valid]))
        dmax = float(np.nanmax(dem[center_valid]))
    else:
        dmin, dmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    dem_norm = np.full_like(dem, np.nan)
    dem_norm[center_valid] = (dem[center_valid] - dmin) / max(dmax - dmin, 1e-6)
    thresh_norm = wall_pct / 100.0
    if use_floor:
        wr, wc = np.where(center_valid & (dem_norm <= thresh_norm))
    else:
        wr, wc = np.where(center_valid & (dem_norm >= thresh_norm))

    # Pixel (row, col) → UTM → PLY local
    utm_x   = dem_utm_x0   + wc * gt[1]   # left edge of cell
    utm_y   = dem_utm_y_top + wr * gt[5]  # top edge of cell (gt[5] < 0 → decreasing)
    local_x = utm_x - offset_x
    local_y = utm_y - offset_y

    wall_pts = np.stack([local_x, local_y], axis=1)
    thresh_abs = dmin + thresh_norm * (dmax - dmin)
    direction = "<=" if use_floor else ">="
    label = "floor" if use_floor else "wall-top"
    print(f"    {len(wall_pts)} {label} cells (center-norm{direction}{thresh_norm:.2f}, "
          f"center range [{dmin:.3f},{dmax:.3f}] m, thresh={thresh_abs:.3f} m abs)")
    print(f"    Local extent: X=[{local_x.min():.2f},{local_x.max():.2f}] "
          f"Y=[{local_y.min():.2f},{local_y.max():.2f}]")

    # Debug image: normalized grayscale DEM (0→black, 1→white) + wall-top in red
    # dem_norm is NaN outside center region; treat as 0 for display
    img8 = (np.nan_to_num(dem_norm, nan=0.0) * 255).astype(np.uint8)
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
# Experiment 1: Circle-based DEM PCA
# ---------------------------------------------------------------------------

def _circle_floor_pts_lidar(
    lidar_pts: np.ndarray,
    cell_size: float = 0.02,
    pct: float = 60,
) -> tuple:
    """
    Build LiDAR DEM (max-Y per XZ cell) and select the lowest pct-th data
    percentile within a circle centred at the DEM grid centre.
    Circle diameter = shorter physical dimension / 2.

    Returns (pts_xz, dem, (x0,z0), debug_bgr)
    """
    x   = lidar_pts[:, 0].astype(np.float32)
    z   = lidar_pts[:, 2].astype(np.float32)
    elv = lidar_pts[:, 1].astype(np.float32)

    x0, z0 = float(x.min()), float(z.min())
    col = ((x - x0) / cell_size).astype(np.int32)
    row = ((z - z0) / cell_size).astype(np.int32)
    W, H = int(col.max()) + 1, int(row.max()) + 1

    flat = np.full(H * W, -np.inf, dtype=np.float32)
    np.maximum.at(flat, row * W + col, elv)
    dem = flat.reshape(H, W)
    valid = dem > -np.inf
    dem[~valid] = np.nan

    # Circle: diameter = shorter physical side / 2  →  radius = shorter / 4
    shorter_m  = min(H, W) * cell_size
    radius_m   = shorter_m / 4.0
    radius_px  = radius_m / cell_size          # same in both axes (square cells)
    cr, cc_c   = H / 2.0, W / 2.0
    rr, cc_g   = np.mgrid[0:H, 0:W]
    circle     = ((rr - cr)**2 + (cc_g - cc_c)**2) <= radius_px**2

    in_circle = valid & circle
    if not in_circle.any():
        return np.empty((0, 2)), dem, (x0, z0), None

    thresh = float(np.percentile(dem[in_circle], pct))
    wr, wc = np.where(in_circle & (dem <= thresh))
    pts = np.stack([x0 + wc * cell_size, z0 + wr * cell_size], axis=1)

    print(f"  LiDAR circle: center=({cr:.0f},{cc_c:.0f}) "
          f"r={radius_px:.0f} cells ({radius_m:.2f}m), "
          f"{len(pts)} floor pts (p{pct:.0f}={thresh:.3f})")

    dmin, dmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    img8 = ((dem - dmin) / max(dmax - dmin, 1e-6) * 255).astype(np.uint8)
    img8 = np.nan_to_num(img8, nan=0).astype(np.uint8)
    debug = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    debug[wr, wc] = (0, 0, 255)
    cv2.circle(debug, (int(cc_c), int(cr)), int(radius_px), (0, 255, 0), 2)

    return pts, dem, (x0, z0), debug


def _circle_floor_pts_geotiff(
    dem_path: str,
    ply_world_bbox: tuple,
    pct: float = 60,
) -> tuple:
    """
    Load GeoTIFF DEM and select the lowest pct-th data percentile within a
    circle centred at the DEM grid centre.
    Circle diameter = shorter physical dimension / 2.

    Returns (pts_local_xy, debug_bgr).
    """
    from osgeo import gdal
    gdal.UseExceptions()

    ds  = gdal.Open(dem_path)
    gt  = ds.GetGeoTransform()
    W_d, H_d = ds.RasterXSize, ds.RasterYSize
    band   = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem    = band.ReadAsArray().astype(np.float32)
    if nodata is not None:
        dem[dem == nodata] = np.nan

    cell_x = abs(float(gt[1]))
    cell_y = abs(float(gt[5]))

    dem_utm_x0    = gt[0]
    dem_utm_y_top = gt[3]
    dem_utm_y_bot = gt[3] + gt[5] * H_d

    ply_x0, ply_y0 = ply_world_bbox[:2]
    offset_x = dem_utm_x0 - ply_x0
    offset_y = dem_utm_y_bot - ply_y0

    # Ellipse mask (handles non-square pixels)
    phys_h = H_d * cell_y
    phys_w = W_d * cell_x
    shorter_m  = min(phys_h, phys_w)
    radius_m   = shorter_m / 4.0
    radius_r   = radius_m / cell_y
    radius_c   = radius_m / cell_x
    cr, cc_c   = H_d / 2.0, W_d / 2.0
    rr, cc_g   = np.mgrid[0:H_d, 0:W_d]
    circle     = (((rr - cr) / radius_r)**2 + ((cc_g - cc_c) / radius_c)**2) <= 1.0

    valid     = ~np.isnan(dem)
    in_circle = valid & circle
    if not in_circle.any():
        return np.empty((0, 2)), None

    thresh = float(np.percentile(dem[in_circle], pct))
    wr, wc = np.where(in_circle & (dem <= thresh))

    utm_x   = dem_utm_x0    + wc * gt[1]
    utm_y   = dem_utm_y_top + wr * gt[5]
    local_x = utm_x - offset_x
    local_y = utm_y - offset_y
    pts = np.stack([local_x, local_y], axis=1)

    print(f"  GeoTIFF circle: center=({cr:.0f},{cc_c:.0f}) "
          f"r=({radius_r:.0f}r,{radius_c:.0f}c) ({radius_m:.2f}m), "
          f"{len(pts)} floor pts (p{pct:.0f}={thresh:.3f}m)")

    dmin, dmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    img8 = ((dem - dmin) / max(dmax - dmin, 1e-6) * 255).astype(np.uint8)
    img8 = np.nan_to_num(img8, nan=0).astype(np.uint8)
    debug = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    debug[wr, wc] = (0, 0, 255)
    cv2.ellipse(debug, (int(cc_c), int(cr)),
                (int(radius_c), int(radius_r)), 0, 0, 360, (0, 255, 0), 2)

    return pts, debug


def register_lidar_to_ply_world_circle(
    lidar_pts: np.ndarray,
    lidar_xz_bbox: tuple,
    dem_path: str,
    ply_world_bbox: tuple,
    lidar_render: np.ndarray,
    ply_render: np.ndarray,
    xz_polygon: np.ndarray,
    lidar_cell_size: float = 0.02,
    pct: float = 60,
) -> tuple:
    """
    Circle-based DEM PCA registration.

    Selects the lowest pct-th data percentile (not range fraction) within a
    centred circle (diameter = shorter DEM dimension / 2) from both the LiDAR
    DEM and the GeoTIFF DEM, then runs PCA alignment (scale=1.0).

    Returns (transform_fn, debug_img, note, dem_debug).
    dem_debug = {'lidar_dem_img': ..., 'ply_dem_img': ...}
    """
    lidar_wall, _, _, lidar_debug = _circle_floor_pts_lidar(
        lidar_pts, cell_size=lidar_cell_size, pct=pct)
    if len(lidar_wall) < 10:
        raise RuntimeError(f"Circle reg: too few LiDAR floor cells ({len(lidar_wall)})")

    ply_wall, ply_debug = _circle_floor_pts_geotiff(dem_path, ply_world_bbox, pct=pct)
    if len(ply_wall) < 10:
        raise RuntimeError(f"Circle reg: too few GeoTIFF floor cells ({len(ply_wall)})")

    def _pca2(pts):
        center = pts.mean(axis=0)
        cov    = np.cov(pts.T)
        eigval, eigvec = np.linalg.eigh(cov)
        main  = eigvec[:, -1]
        angle = float(np.degrees(np.arctan2(main[1], main[0])))
        return center, angle, float(np.sqrt(eigval[-1])), float(np.sqrt(eigval[0]))

    cx_li, ang_li, sml_li, spr_li = _pca2(lidar_wall)
    cx_pl, ang_pl, sml_pl, spr_pl = _pca2(ply_wall)
    scale = 1.0

    print(f"  Circle PCA LiDAR XZ: center=({cx_li[0]:.3f},{cx_li[1]:.3f}) "
          f"angle={ang_li:.1f}° main={sml_li:.3f}m perp={spr_li:.3f}m")
    print(f"  Circle PCA PLY XY:   center=({cx_pl[0]:.3f},{cx_pl[1]:.3f}) "
          f"angle={ang_pl:.1f}° main={sml_pl:.3f}m perp={spr_pl:.3f}m")

    def _make_M(rot_deg):
        r = np.radians(rot_deg)
        c, s = np.cos(r), np.sin(r)
        tx = cx_pl[0] - scale * (c * cx_li[0] - s * cx_li[1])
        ty = cx_pl[1] - scale * (s * cx_li[0] + c * cx_li[1])
        return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=float)

    def _apply(M, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M @ h.T).T[:, :2]

    px0, py0, px1, py1 = ply_world_bbox

    def _within_bounds(wpts, margin=0.5):
        ex = (px1 - px0) * margin
        ey = (py1 - py0) * margin
        return (wpts[:, 0].min() > px0 - ex and wpts[:, 0].max() < px1 + ex and
                wpts[:, 1].min() > py0 - ey and wpts[:, 1].max() < py1 + ey)

    rot_deg = ang_pl - ang_li
    chosen_M, note = None, ""
    for rotation in [rot_deg, rot_deg + 180]:
        M = _make_M(rotation)
        if _within_bounds(_apply(M, xz_polygon)):
            chosen_M, note = M, f"Circle PCA rotation={rotation:.1f}°"
            break
    if chosen_M is None:
        chosen_M = _make_M(rot_deg)
        note = f"Circle PCA rotation={rot_deg:.1f}° (fallback)"
    print(f"  {note}")

    def transform_fn(xz_pts):
        return _apply(chosen_M, xz_pts)

    pH, pW = ply_render.shape[:2]
    world_poly = _apply(chosen_M, xz_polygon)
    ppx = np.clip(((world_poly[:, 0] - px0) / (px1 - px0) * pW).astype(np.int32), 0, pW - 1)
    ppy = np.clip(((py1 - world_poly[:, 1]) / (py1 - py0) * pH).astype(np.int32), 0, pH - 1)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [np.stack([ppx, ppy], axis=1).reshape(-1, 1, 2)],
                  isClosed=True, color=(255, 0, 255), thickness=4)

    return transform_fn, debug_img, note, {"lidar_dem_img": lidar_debug, "ply_dem_img": ply_debug}


# ---------------------------------------------------------------------------
# Experiment 2: ICP refinement after RGB footprint PCA
# ---------------------------------------------------------------------------

def register_lidar_to_ply_world_icp(
    lidar_pts: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_cloud,                   # CloudComPy ccPointCloud object
    ply_world_bbox: tuple,
    lidar_render: np.ndarray,
    ply_render: np.ndarray,
    xz_polygon: np.ndarray,
) -> tuple:
    """
    ICP-based registration.

    1. RGB footprint PCA gives an initial LiDAR XZ → PLY world XY transform.
    2. The transformed LiDAR XZ points (Z=0 padded) and the PLY cloud (Z=0
       projected) are fed into CloudComPy ICP in 2D (horizontal only: only
       Z-axis rotation + XY translation are allowed).
    3. The ICP refinement is composed with the PCA initial transform.

    Returns (transform_fn, debug_img, note, icp_debug).
    icp_debug = {'initial_pca_img': ...}
    """
    import cloudComPy as cc

    # ---- Step 1: initial RGB PCA alignment ----
    transform_pca, debug_pca, note_pca, _ = register_lidar_to_ply_world(
        lidar_render, lidar_xz_bbox, ply_render, ply_world_bbox, xz_polygon)
    print(f"  ICP initial alignment: {note_pca}")

    # ---- Step 2: apply PCA to all LiDAR XZ → approx PLY XY ----
    lidar_xz = lidar_pts[:, [0, 2]].astype(np.float32)
    approx_xy = transform_pca(lidar_xz).astype(np.float64)

    rng = np.random.default_rng(0)
    if len(approx_xy) > 50000:
        idx = rng.choice(len(approx_xy), 50000, replace=False)
        approx_xy = approx_xy[idx]

    lidar_3d = np.column_stack([approx_xy, np.zeros(len(approx_xy))])
    lidar_cc = cc.ccPointCloud("lidar_2d")
    lidar_cc.coordsFromNPArray_copy(lidar_3d)

    # ---- Step 3: PLY cloud projected to Z=0 ----
    ply_pts_all = ply_cloud.toNpArrayCopy()[:, :2].astype(np.float64)
    if len(ply_pts_all) > 100000:
        idx_p = rng.choice(len(ply_pts_all), 100000, replace=False)
        ply_pts_all = ply_pts_all[idx_p]
    ply_3d = np.column_stack([ply_pts_all, np.zeros(len(ply_pts_all))])
    ply_cc = cc.ccPointCloud("ply_2d")
    ply_cc.coordsFromNPArray_copy(ply_3d)

    print(f"  ICP: {len(lidar_3d)} LiDAR pts vs {len(ply_3d)} PLY pts (2D projected)")

    # ---- Step 4: run ICP — horizontal only ----
    # transformationFilters: SKIP_RYZ=2 (no X-axis rotation),
    #   SKIP_RXZ=4 (no Y-axis rotation), SKIP_TZ=32 (no Z translation)
    result = cc.ICP(
        data=lidar_cc, model=ply_cc,
        minRMSDecrease=1e-6,
        maxIterationCount=50,
        randomSamplingLimit=50000,
        removeFarthestPoints=True,
        method=cc.CONVERGENCE_TYPE.MAX_ITER_CONVERGENCE,
        finalOverlapRatio=0.8,
        adjustScale=False,
        transformationFilters=2 | 4 | 32,
    )

    # ---- Step 5: extract 2D transform from 4×4 ICP matrix ----
    # ccGLMatrix.data() returns 16 floats in column-major (OpenGL) order;
    # reshape with order='F' (Fortran/column-major) gives row-major 4×4.
    mat = np.array(result.transMat.data()).reshape(4, 4, order='F')

    print(f"  ICP RMS={result.finalRMS:.5f}  pts={result.finalPointCount}")
    print(f"  ICP matrix (row-major):\n{np.round(mat, 4)}")

    angle_icp = float(np.arctan2(mat[1, 0], mat[0, 0]))
    tx_icp    = float(mat[0, 3])
    ty_icp    = float(mat[1, 3])
    print(f"  ICP refinement: Δangle={np.degrees(angle_icp):.2f}°  "
          f"Δtx={tx_icp:.3f}m  Δty={ty_icp:.3f}m")

    cos_a, sin_a = np.cos(angle_icp), np.sin(angle_icp)
    R_icp = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    t_icp = np.array([tx_icp, ty_icp])

    def transform_fn(xz_pts: np.ndarray) -> np.ndarray:
        xy_approx = transform_pca(xz_pts.astype(np.float32))
        return (R_icp @ xy_approx.T).T + t_icp

    # ---- Step 6: debug image ----
    px0, py0, px1, py1 = ply_world_bbox
    pH, pW = ply_render.shape[:2]
    world_poly = transform_fn(xz_polygon)
    ppx = np.clip(((world_poly[:, 0] - px0) / (px1 - px0) * pW).astype(np.int32), 0, pW - 1)
    ppy = np.clip(((py1 - world_poly[:, 1]) / (py1 - py0) * pH).astype(np.int32), 0, pH - 1)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [np.stack([ppx, ppy], axis=1).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 200, 255), thickness=4)   # amber for ICP

    note = (f"ICP (PCA init: {note_pca} | "
            f"refine Δ{np.degrees(angle_icp):.1f}° tx={tx_icp:.2f} ty={ty_icp:.2f} "
            f"RMS={result.finalRMS:.4f})")

    return transform_fn, debug_img, note, {"initial_pca_img": debug_pca}


# ---------------------------------------------------------------------------
# Experiment 3: 3-D ICP refinement after RGB footprint PCA
# ---------------------------------------------------------------------------

def register_lidar_to_ply_world_icp_3d(
    lidar_pts: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_cloud,
    ply_world_bbox: tuple,
    lidar_render: np.ndarray,
    ply_render: np.ndarray,
    xz_polygon: np.ndarray,
) -> tuple:
    """
    3-D ICP refinement of an RGB footprint PCA initial alignment.

    Unlike the 2-D version, this builds proper 3-D point clouds:
      - LiDAR: (X_pl, Y_pl) from 2-D PCA + LiDAR Y (elevation) shifted by the
        estimated elevation offset between the two datasets.
      - PLY model: full 3-D surface cloud (X, Y, Z).

    ICP is constrained to rotation around Z only + full 3-D translation
    (transformationFilters = SKIP_RYZ | SKIP_RXZ = 6), so the scan cannot
    tilt. The horizontal part of the refined transform is then composed with
    the PCA initial transform to produce the final 2-D polygon mapping.

    Returns (transform_fn, debug_img, note, icp_debug).
    icp_debug = {'initial_pca_img': ...}
    """
    import cloudComPy as cc

    # ---- Step 1: initial RGB PCA ----
    transform_pca, debug_pca, note_pca, _ = register_lidar_to_ply_world(
        lidar_render, lidar_xz_bbox, ply_render, ply_world_bbox, xz_polygon)
    print(f"  3-D ICP initial alignment: {note_pca}")

    # ---- Step 2: elevation offset estimation ----
    ply_pts_all = ply_cloud.toNpArrayCopy()[:, :3].astype(np.float64)
    ply_z_mean  = float(np.mean(ply_pts_all[:, 2]))
    lidar_y_mean = float(np.mean(lidar_pts[:, 1]))
    elev_offset = ply_z_mean - lidar_y_mean
    print(f"  PLY Z range: [{ply_pts_all[:,2].min():.2f}, {ply_pts_all[:,2].max():.2f}]  mean={ply_z_mean:.3f}")
    print(f"  LiDAR Y range: [{lidar_pts[:,1].min():.2f}, {lidar_pts[:,1].max():.2f}]  mean={lidar_y_mean:.3f}")
    print(f"  Estimated elevation offset: {elev_offset:.3f} m")

    # ---- Step 3: build LiDAR 3-D cloud ----
    lidar_xz = lidar_pts[:, [0, 2]].astype(np.float32)
    approx_xy = transform_pca(lidar_xz).astype(np.float64)
    lidar_z   = lidar_pts[:, 1].astype(np.float64) + elev_offset

    lidar_3d = np.column_stack([approx_xy, lidar_z])
    rng = np.random.default_rng(0)
    if len(lidar_3d) > 50000:
        lidar_3d = lidar_3d[rng.choice(len(lidar_3d), 50000, replace=False)]

    lidar_cc = cc.ccPointCloud("lidar_3d")
    lidar_cc.coordsFromNPArray_copy(lidar_3d)

    # ---- Step 4: PLY model cloud ----
    if len(ply_pts_all) > 100000:
        ply_pts_all = ply_pts_all[rng.choice(len(ply_pts_all), 100000, replace=False)]

    ply_cc = cc.ccPointCloud("ply_3d")
    ply_cc.coordsFromNPArray_copy(ply_pts_all)

    print(f"  3-D ICP: {len(lidar_3d)} LiDAR pts vs {len(ply_pts_all)} PLY pts")

    # ---- Step 5: run 3-D ICP (horizontal rotation + full 3-D translation) ----
    # SKIP_RYZ=2 (no rotation around X), SKIP_RXZ=4 (no rotation around Y)
    result = cc.ICP(
        data=lidar_cc, model=ply_cc,
        minRMSDecrease=1e-6,
        maxIterationCount=50,
        randomSamplingLimit=50000,
        removeFarthestPoints=True,
        method=cc.CONVERGENCE_TYPE.MAX_ITER_CONVERGENCE,
        finalOverlapRatio=0.5,
        adjustScale=False,
        transformationFilters=2 | 4,
    )

    mat = np.array(result.transMat.data()).reshape(4, 4, order='F')
    angle_icp = float(np.arctan2(mat[1, 0], mat[0, 0]))
    tx_icp    = float(mat[0, 3])
    ty_icp    = float(mat[1, 3])
    tz_icp    = float(mat[2, 3])

    print(f"  3-D ICP RMS={result.finalRMS:.5f}  pts={result.finalPointCount}")
    print(f"  3-D ICP refinement: Δangle={np.degrees(angle_icp):.2f}°  "
          f"Δtx={tx_icp:.3f}m  Δty={ty_icp:.3f}m  Δtz={tz_icp:.3f}m")

    cos_a, sin_a = np.cos(angle_icp), np.sin(angle_icp)
    R_icp = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    t_icp = np.array([tx_icp, ty_icp])

    def transform_fn(xz_pts: np.ndarray) -> np.ndarray:
        xy_approx = transform_pca(xz_pts.astype(np.float32))
        return (R_icp @ xy_approx.T).T + t_icp

    # ---- Step 6: debug image ----
    px0, py0, px1, py1 = ply_world_bbox
    pH, pW = ply_render.shape[:2]
    world_poly = transform_fn(xz_polygon)
    ppx = np.clip(((world_poly[:, 0] - px0) / (px1 - px0) * pW).astype(np.int32), 0, pW - 1)
    ppy = np.clip(((py1 - world_poly[:, 1]) / (py1 - py0) * pH).astype(np.int32), 0, pH - 1)
    debug_img = ply_render.copy()
    cv2.polylines(debug_img, [np.stack([ppx, ppy], axis=1).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 128, 255), thickness=4)   # orange for 3-D ICP

    note = (f"3-D ICP (PCA init: {note_pca} | "
            f"Δ{np.degrees(angle_icp):.1f}° tx={tx_icp:.2f} ty={ty_icp:.2f} tz={tz_icp:.2f} "
            f"RMS={result.finalRMS:.4f})")

    return transform_fn, debug_img, note, {"initial_pca_img": debug_pca}


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def save_lidar_debug(
    lidar_render: np.ndarray,
    xz_polygon: np.ndarray,
    lidar_xz_bbox: tuple,
    out_path: str,
    xz_polygons: list = None,
) -> None:
    """Save LiDAR top-down render with annotation polygons overlaid.

    Largest polygon (xz_polygon, used for registration) drawn in green.
    Secondary polygons from xz_polygons drawn in cyan so multi-polygon
    detection is visually verifiable.
    """
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    lH, lW = lidar_render.shape[:2]
    debug = lidar_render.copy()

    def _to_px(poly):
        px = ((poly[:, 0] - lx0) / (lx1 - lx0) * lW).astype(np.int32)
        pz = ((poly[:, 1] - lz0) / (lz1 - lz0) * lH).astype(np.int32)
        return np.stack([px, pz], axis=1).reshape(-1, 1, 2)

    # Secondary polygons in cyan first (drawn underneath)
    if xz_polygons is not None:
        for poly in xz_polygons:
            if not np.array_equal(poly, xz_polygon):
                cv2.polylines(debug, [_to_px(poly)], isClosed=True,
                              color=(255, 255, 0), thickness=3)

    # Largest (registration) polygon in green on top
    cv2.polylines(debug, [_to_px(xz_polygon)], isClosed=True,
                  color=(0, 255, 0), thickness=4)
    cv2.imwrite(out_path, debug)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Edge-based feature matching registration: LiDAR render → PLY render → PLY world
#
# Insight: both renders show the same stone wall edges even though their textures
# differ (LiDAR = baked UV mesh, PLY = photogrammetry RGB). Canny edge detection
# isolates the structural wall/surface boundaries that appear in both modalities.
# SIFT on edge images finds keypoints at edge junctions/corners rather than
# texture patches — more modality-invariant than raw SIFT on the renders.
#
# Pipeline:
#   1. Canny edge detection on both renders (grayscale)
#   2. Optional dilation to help matching
#   3. SIFT keypoint extraction + FLANN matching + Lowe ratio test
#   4. RANSAC affine partial 2D (estimateAffinePartial2D) with reprojThreshold=8px
#   5. Force scale=1.0; recompute translation anchored at inlier centroid pair
#   6. 180° ambiguity check via yellow polygon location
#   7. Return (transform_fn, debug_img, note) matching other registration functions
# ---------------------------------------------------------------------------

def register_lidar_to_ply_world_edges(
    lidar_render: np.ndarray,
    lidar_xz_bbox: tuple,
    ply_render: np.ndarray,
    ply_world_bbox: tuple,
    xz_polygon: np.ndarray,
    su_number: str = "unknown",
    output_dir: str = None,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_px: int = 2,
    lowe_ratio: float = 0.75,
    ransac_reproj: float = 8.0,
    min_inliers: int = 8,
) -> tuple:
    """
    Edge-based feature matching: LiDAR render → PLY render → PLY world XY.

    Converts both top-down renders to Canny edge maps, runs SIFT feature
    matching on the edge images (finding wall corner/junction features rather
    than texture patches), estimates rotation+translation via RANSAC affine
    partial 2D, forces scale=1.0, resolves 180° ambiguity, and returns the
    same interface as other registration functions.

    Args:
        lidar_render:   BGR top-down render of LiDAR mesh (from process_usdz)
        lidar_xz_bbox:  (x0, z0, x1, z1) LiDAR world coords for the render
        ply_render:     BGR top-down render of PLY point cloud
        ply_world_bbox: (x0, y0, x1, y1) PLY world coords for the render
        xz_polygon:     (M, 2) yellow annotation polygon in LiDAR XZ space
        su_number:      SU identifier string (for debug image filenames)
        output_dir:     if set, edge debug images are saved here
        canny_low:      Canny lower threshold (default 50)
        canny_high:     Canny upper threshold (default 150)
        dilate_px:      dilation kernel radius in pixels (0 = no dilation)
        lowe_ratio:     Lowe's ratio test threshold (default 0.75)
        ransac_reproj:  RANSAC reprojection threshold in pixels (default 8)
        min_inliers:    minimum RANSAC inliers required (default 8)

    Returns:
        transform_fn: callable (N,2) LiDAR XZ → (N,2) PLY world XY
        debug_img:    BGR image: PLY render with cyan inlier keypoints + magenta polygon
        note:         str describing the result (rotation, inlier count, etc.)

    Raises:
        RuntimeError: if fewer than min_inliers RANSAC inliers are found
    """
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox

    # ------------------------------------------------------------------
    # 1. Edge detection on both renders
    # ------------------------------------------------------------------
    lidar_gray = cv2.cvtColor(lidar_render, cv2.COLOR_BGR2GRAY)
    ply_gray   = cv2.cvtColor(ply_render,   cv2.COLOR_BGR2GRAY)

    lidar_edges = cv2.Canny(lidar_gray, canny_low, canny_high)
    ply_edges   = cv2.Canny(ply_gray,   canny_low, canny_high)

    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        lidar_edges = cv2.dilate(lidar_edges, kernel)
        ply_edges   = cv2.dilate(ply_edges,   kernel)

    print(f"  Edge pixels -- LiDAR: {lidar_edges.sum() // 255}  PLY: {ply_edges.sum() // 255}")

    # Save edge debug images if requested
    if output_dir is not None:
        lidar_edge_path = os.path.join(output_dir, f"debug_SU{su_number}_lidar_edges.png")
        ply_edge_path   = os.path.join(output_dir, f"debug_SU{su_number}_ply_edges.png")
        cv2.imwrite(lidar_edge_path, lidar_edges)
        cv2.imwrite(ply_edge_path,   ply_edges)
        print(f"  Saved edge images: {lidar_edge_path}")
        print(f"                     {ply_edge_path}")

    # ------------------------------------------------------------------
    # 2. SIFT on edge images
    # ------------------------------------------------------------------
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(lidar_edges, None)
    kp_p, des_p = sift.detectAndCompute(ply_edges,   None)

    print(f"  SIFT keypoints -- LiDAR: {len(kp_l) if kp_l else 0}  PLY: {len(kp_p) if kp_p else 0}")

    if des_l is None or des_p is None or len(kp_l) < 4 or len(kp_p) < 4:
        raise RuntimeError(
            f"Edge-SIFT: too few keypoints (LiDAR={len(kp_l) if kp_l else 0}, "
            f"PLY={len(kp_p) if kp_p else 0}). "
            "Try lowering canny_low or reducing dilate_px."
        )

    # ------------------------------------------------------------------
    # 3. FLANN matching + Lowe's ratio test
    # ------------------------------------------------------------------
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(des_l, des_p, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < lowe_ratio * n.distance:
                good.append(m)

    print(f"  FLANN+Lowe matches: {len(good)}")

    if len(good) < 4:
        raise RuntimeError(
            f"Edge-SIFT: too few Lowe-filtered matches ({len(good)}). "
            "Try adjusting canny thresholds or lowe_ratio."
        )

    # ------------------------------------------------------------------
    # 4. RANSAC affine partial 2D (rotation + translation + scale)
    # ------------------------------------------------------------------
    src_pts = np.float32([kp_l[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_p[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_affine, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj,
    )

    if M_affine is None or inlier_mask is None:
        raise RuntimeError("Edge-SIFT: RANSAC failed (estimateAffinePartial2D returned None)")

    n_inliers = int(inlier_mask.sum())
    print(f"  RANSAC inliers: {n_inliers} / {len(good)}")

    if n_inliers < min_inliers:
        raise RuntimeError(
            f"Edge-SIFT: only {n_inliers} RANSAC inliers (need {min_inliers}). "
            "Increase ransac_reproj or lower canny thresholds for denser edges."
        )

    # ------------------------------------------------------------------
    # 5. Extract rotation; force scale=1.0; recompute translation
    #    from inlier centroid pair
    # ------------------------------------------------------------------
    # M_affine = [[a, -b, tx], [b, a, ty]] where (a,b) encode rotation+scale
    a_val   = float(M_affine[0, 0])
    b_val   = float(M_affine[1, 0])
    rot_rad = np.arctan2(b_val, a_val)
    rot_deg = float(np.degrees(rot_rad))
    print(f"  RANSAC rotation (raw, before scale=1 override): {rot_deg:.2f} deg")

    inlier_idx  = np.where(inlier_mask.ravel() == 1)[0]
    src_inliers = src_pts[inlier_idx, 0, :]   # (K, 2)
    dst_inliers = dst_pts[inlier_idx, 0, :]

    def _build_M_scale1(angle_deg, src_ctr, dst_ctr):
        """Build 2x3 affine matrix: scale=1, rotate angle_deg, translate to match centroids."""
        r  = np.radians(angle_deg)
        c, s = np.cos(r), np.sin(r)
        R2 = np.array([[c, -s], [s, c]])
        t  = dst_ctr - R2 @ src_ctr
        return np.hstack([R2, t.reshape(2, 1)])   # (2, 3)

    src_ctr = src_inliers.mean(axis=0)
    dst_ctr = dst_inliers.mean(axis=0)
    M_scale1 = _build_M_scale1(rot_deg, src_ctr, dst_ctr)

    # ------------------------------------------------------------------
    # 6. 180 deg ambiguity: try both orientations, pick the one that puts
    #    the yellow polygon inside the PLY render bounds
    # ------------------------------------------------------------------
    def _xz_to_lidar_px(xz_pts):
        col = (xz_pts[:, 0] - lx0) / (lx1 - lx0) * lW
        row = (xz_pts[:, 1] - lz0) / (lz1 - lz0) * lH
        return np.stack([col, row], axis=1)

    def _apply_affine(M23, pts):
        """Apply 2x3 affine matrix to (N,2) points."""
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M23 @ h.T).T   # (N, 2)

    def _ply_px_to_world(ply_px):
        wx = px0 + ply_px[:, 0] * (px1 - px0) / pW
        wy = py1 - ply_px[:, 1] * (py1 - py0) / pH
        return np.stack([wx, wy], axis=1)

    def _within_ply_render(ply_px, margin=0.20):
        return (ply_px[:, 0].min() > -pW * margin and
                ply_px[:, 0].max() <  pW * (1 + margin) and
                ply_px[:, 1].min() > -pH * margin and
                ply_px[:, 1].max() <  pH * (1 + margin))

    yellow_lidar_px = _xz_to_lidar_px(xz_polygon)

    chosen_M23, note = None, ""
    for flip in [0, 180]:
        angle_try = rot_deg + flip
        M_try = _build_M_scale1(angle_try, src_ctr, dst_ctr)
        ply_px_try = _apply_affine(M_try, yellow_lidar_px)
        if _within_ply_render(ply_px_try):
            chosen_M23 = M_try
            note = (f"Edge-SIFT rotation={angle_try:.1f} deg "
                    f"scale=1.0 inliers={n_inliers}")
            print(f"  180-deg check: flip={flip} deg -> polygon within bounds -> chosen")
            break
        else:
            print(f"  180-deg check: flip={flip} deg -> polygon outside bounds, trying next")

    if chosen_M23 is None:
        chosen_M23 = M_scale1
        note = (f"Edge-SIFT rotation={rot_deg:.1f} deg scale=1.0 inliers={n_inliers} "
                f"(fallback -- polygon outside PLY render bounds)")
        print(f"  {note}")
    else:
        print(f"  {note}")

    def transform_fn(xz_pts: np.ndarray) -> np.ndarray:
        """LiDAR XZ -> PLY world XY."""
        lidar_px = _xz_to_lidar_px(xz_pts)
        ply_px   = _apply_affine(chosen_M23, lidar_px)
        return _ply_px_to_world(ply_px)

    # ------------------------------------------------------------------
    # 7. Debug image: PLY render + cyan inlier keypoints + magenta polygon
    # ------------------------------------------------------------------
    debug_img = ply_render.copy()

    # Draw inlier destination keypoints in cyan
    for pt in dst_inliers:
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)

    # Draw transformed yellow polygon in magenta
    world_poly = transform_fn(xz_polygon)
    ppx = ((world_poly[:, 0] - px0) / (px1 - px0) * pW).astype(np.int32)
    ppy = ((py1 - world_poly[:, 1]) / (py1 - py0) * pH).astype(np.int32)
    cv2.polylines(debug_img,
                  [np.stack([ppx, ppy], axis=1).reshape(-1, 1, 2)],
                  isClosed=True, color=(255, 0, 255), thickness=4)

    return transform_fn, debug_img, note
