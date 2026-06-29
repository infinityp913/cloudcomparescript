#!/usr/bin/env python3
"""
eval_methods.py — Evaluate all registration methods against annotated ground truth.

Run: ./run.sh eval_methods.py
Output: Data/eval/<site>_<method>.png (debug composites), Data/eval/ranking.txt

GT is extracted from diff(annotated_ortho, raw_ortho) via:
  absdiff > 20 → morph-close → fill interior → outer contour → GT polygon(s)
GT polygon(s) are mapped from ortho pixel space to PLY render pixel space via
content-bbox normalisation (non-black region of raw ortho ↔ PLY world bbox).

Metrics per method×site:
  centroid_err_m — distance between GT centroid and placed polygon centroid (metres)
  iou            — intersection-over-union of filled polygon masks
"""

import cloudComPy as cc
import cv2
import numpy as np
import os
import sys
import json
import glob
import base64
import requests

import auto_snip_lidar

cc.initCC()

# Pass --or-only to skip standard + Claude methods and only run OpenRouter models
OR_ONLY = "--or-only" in sys.argv
# Pass --pro-only to run only the gemini-2.5-pro OpenRouter model
PRO_ONLY = "--pro-only" in sys.argv
if PRO_ONLY:
    OR_ONLY = True
# Pass --new-only to run only the new OpenRouter models (skip existing gemini/gpt4o)
NEW_ONLY = "--new-only" in sys.argv
if NEW_ONLY:
    OR_ONLY = True

# Few-shot example site: 20002 is used as the in-context example; test sites = 20003/20005/21001
EXAMPLE_SITE_ID = "20002"

INPUT_MESH_PATH = os.path.expanduser("~/Documents/TARP/ply/")
DATA_DIR = os.path.expanduser("./Data")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Site config ───────────────────────────────────────────────────────────────
SITES = [
    dict(id="20002", json_file="example-20002.json",
         ortho_raw="orthos/ortho_20002_top786.png",
         ortho_ann="orthos/ortho_20002_top786_annotated.png"),
    dict(id="20003", json_file="example-20003.json",
         ortho_raw="orthos/ortho_20003_top786.png",
         ortho_ann="orthos/ortho_20003_top786_annotated.png"),
    dict(id="20005", json_file="example-20005.json",
         ortho_raw="orthos/ortho_20005_top789.png",
         ortho_ann="orthos/ortho_20005_top789_annotated.png"),
    dict(id="21001", json_file="example-21001.json",
         ortho_raw="orthos/ortho_21001_top791.png",
         ortho_ann="orthos/ortho_21001_top791_annotated.png"),
]

# ── PLY helpers ───────────────────────────────────────────────────────────────
def find_mesh_by_pgram_job(job_number):
    pattern = f"Pgram_Job_{job_number}"
    for fname in os.listdir(INPUT_MESH_PATH):
        if fname.endswith(".ply") and pattern in fname:
            return fname.replace(".ply", "")
    return None


def find_top_bin(top_id):
    search_dir = os.path.join(DATA_DIR, top_id)
    for b in glob.glob(os.path.join(search_dir, "*.bin")):
        if "top_with_dist" in os.path.basename(b).lower():
            return b
    raise FileNotFoundError(f"No *top_with_dist*.bin found in {search_dir}")


def render_topdown(cloud, resolution=0.01):
    coords = cloud.toNpArrayCopy()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0, y0, x1, y1 = float(x.min()), float(y.min()), float(x.max()), float(y.max())
    W = int(np.ceil((x1 - x0) / resolution)) + 1
    H = int(np.ceil((y1 - y0) / resolution)) + 1
    print(f"  PLY render: {W}×{H} px  world=({x0:.1f},{y0:.1f})–({x1:.1f},{y1:.1f})")
    img = np.zeros((H, W, 3), dtype=np.uint8)
    px = np.clip(((x - x0) / resolution).astype(int), 0, W - 1)
    py = np.clip((H - 1 - (y - y0) / resolution).astype(int), 0, H - 1)
    order = np.argsort(z)
    px_s, py_s = px[order], py[order]
    if cloud.hasColors():
        try:
            rgba = cloud.colorsToNpArrayCopy()
            img[py_s, px_s] = rgba[order, :3][:, ::-1]
        except Exception:
            pass
    if img.max() == 0:
        z_s = z[order]
        z_norm = ((z_s - z_s.min()) / max(z_s.max() - z_s.min(), 1e-6) * 255).astype(np.uint8)
        img[py_s, px_s] = np.stack([z_norm] * 3, axis=1)
    img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    return img, (x0, y0, x1, y1)


def world_to_render_px(world_pts, world_bbox, render_shape):
    """PLY world (x, y) → PLY render pixel (col, row)."""
    wx0, wy0, wx1, wy1 = world_bbox
    rH, rW = render_shape[:2]
    r = np.empty((len(world_pts), 2))
    r[:, 0] = (world_pts[:, 0] - wx0) / (wx1 - wx0) * rW
    r[:, 1] = (wy1 - world_pts[:, 1]) / (wy1 - wy0) * rH
    return r


# ── Ground truth extraction ───────────────────────────────────────────────────
def content_bbox(img_bgr, threshold=15):
    """Bounding box of the non-black content region in image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0:
        return 0, 0, img_bgr.shape[1] - 1, img_bgr.shape[0] - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_gt_polygons(ortho_raw_path, ortho_ann_path):
    """
    Detect GT annotation polygon(s) by diffing the annotated vs raw ortho image.
    Annotation marks (black or any colour strokes) differ from the raw photo.
    Returns list of (N, 2) float arrays in ortho pixel coords [(x,y), ...].
    """
    raw = cv2.imread(ortho_raw_path)
    ann = cv2.imread(ortho_ann_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot load: {ortho_raw_path}")
    if ann is None:
        raise FileNotFoundError(f"Cannot load: {ortho_ann_path}")
    if raw.shape != ann.shape:
        ann = cv2.resize(ann, (raw.shape[1], raw.shape[0]))

    diff = cv2.absdiff(raw, ann)
    changed = (diff.max(axis=2) > 20).astype(np.uint8) * 255

    # Close the polygon outline gaps
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE,  np.ones((25, 25), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_DILATE, np.ones((5,  5),  np.uint8))

    # Fill the interior of each closed outline
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(raw.shape[:2], np.uint8)
    for c in contours:
        if cv2.contourArea(c) > 500:
            cv2.fillPoly(filled, [c], 255)

    # Re-find outer contours of the filled shapes
    contours2, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for c in contours2:
        if cv2.contourArea(c) > 2000:
            result.append(c.reshape(-1, 2).astype(float))

    print(f"  GT: {len(result)} polygon(s) from diff "
          f"(changed={changed.sum()//255} px)")
    return result


def map_gt_to_render(gt_ortho_polys, ortho_raw, render_world_bbox, render_shape):
    """
    Map GT polygons from ortho pixel space to PLY render pixel space.
    Uses content-bbox normalisation: non-black region of ortho ↔ PLY world bbox.
    Both images are orthographic top-down views of the same scene; the content
    region in the ortho corresponds to the full extent of the PLY render.
    """
    ox0, oy0, ox1, oy1 = content_bbox(ortho_raw)
    ow = ox1 - ox0
    oh = oy1 - oy0
    rH, rW = render_shape[:2]
    result = []
    for pts in gt_ortho_polys:
        r = np.empty_like(pts)
        r[:, 0] = (pts[:, 0] - ox0) / ow * rW
        r[:, 1] = (pts[:, 1] - oy0) / oh * rH
        result.append(r)
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(gt_polys_render, placed_polys_render, render_shape, resolution=0.01):
    """IoU and centroid error (m) comparing union of GT vs placed polygon sets."""
    H, W = render_shape[:2]
    gt_mask = np.zeros((H, W), np.uint8)
    pl_mask = np.zeros((H, W), np.uint8)
    for p in gt_polys_render:
        pts = np.clip(p.astype(np.int32), [[0, 0]], [[W - 1, H - 1]])
        cv2.fillPoly(gt_mask, [pts], 255)
    for p in placed_polys_render:
        pts = np.clip(p.astype(np.int32), [[0, 0]], [[W - 1, H - 1]])
        cv2.fillPoly(pl_mask, [pts], 255)

    inter = np.logical_and(gt_mask > 0, pl_mask > 0).sum()
    union = np.logical_or( gt_mask > 0, pl_mask > 0).sum()
    iou   = float(inter) / float(max(union, 1))

    gt_ys, gt_xs = np.where(gt_mask > 0)
    pl_ys, pl_xs = np.where(pl_mask > 0)
    if len(gt_xs) == 0 or len(pl_xs) == 0:
        return {"centroid_err_px": float("inf"), "centroid_err_m": float("inf"), "iou": iou}

    err_px = float(np.hypot(gt_xs.mean() - pl_xs.mean(), gt_ys.mean() - pl_ys.mean()))
    return {"centroid_err_px": err_px, "centroid_err_m": err_px * resolution, "iou": iou}


# ── Debug image ───────────────────────────────────────────────────────────────
def make_eval_debug(lidar_disp, lidar_polys_px, ply_render,
                    gt_polys_render, placed_polys_render, label):
    """
    LEFT: LiDAR render with annotation polygon(s) in cyan.
    RIGHT: PLY render with GT polygon(s) in red, placed polygon(s) in green.
    """
    li = lidar_disp.copy()
    for p in lidar_polys_px:
        cv2.polylines(li, [p.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 3)

    pl = ply_render.copy()
    for p in gt_polys_render:
        cv2.polylines(pl, [p.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    for p in placed_polys_render:
        cv2.polylines(pl, [p.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    cv2.putText(pl, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),   5)
    cv2.putText(pl, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if li.shape[0] != pl.shape[0]:
        s = pl.shape[0] / li.shape[0]
        li = cv2.resize(li, (int(li.shape[1] * s), pl.shape[0]))
    return np.hstack([li, pl])


# ── OpenRouter vision (same prompt + composite as Claude Vision, no-chamfer) ──
def _call_openrouter(composite_bgr, left_w, right_w, right_h,
                     ann_frac, n_polys, model):
    """
    Call an OpenRouter vision model with the same spatial prompt used for
    Claude Vision. Returns dict {cx, cy, w, h, reasoning} in right-panel coords.
    """
    import re, json as _j

    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_KEY not set")
    ok, buf = cv2.imencode(".png", composite_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    img_b64 = base64.standard_b64encode(buf.tobytes()).decode()

    poly_desc = (f"{n_polys} separate annotated regions" if n_polys > 1
                 else "one annotated region")
    prompt = (
        "You are a spatial registration expert for archaeology.\n\n"
        "The image has TWO panels side by side, separated by a grey divider:\n"
        f"  LEFT ({left_w}px wide): iPhone LiDAR scan from INSIDE an excavation "
        f"trench. The bright cyan-outlined shape(s) show {poly_desc} "
        "painted with yellow spray-paint on the trench floor/walls.\n"
        f"  RIGHT ({right_w}×{right_h}px): Top-down photogrammetry of the SAME "
        "site — like a drone map looking straight down. No annotation shown.\n\n"
        "KEY FACTS:\n"
        f"- The annotated region fills about {ann_frac:.0%} of the LEFT scan.\n"
        "- CRITICAL: The RIGHT panel shows the ENTIRE archaeological site, which "
        "is typically MUCH larger than the LiDAR scan footprint. The matching "
        "region can be ANYWHERE in the RIGHT panel and may appear much smaller "
        "than it does in LEFT. Do NOT assume any particular location or size.\n"
        "- Scale is NOT guaranteed to match between panels — the PLY may cover "
        "2×–10× more physical area than the LiDAR scan.\n"
        "- LEFT shows wall FACES from inside the trench; RIGHT shows wall TOPS "
        "from above.\n"
        "- MATCHING STRATEGY: First look for distinctive internal features inside "
        "the annotated area — square stone blocks, pillars, column bases, hearths. "
        "Find the SAME feature in the RIGHT panel viewed from above. That feature's "
        "location in RIGHT is your best anchor for cx/cy. Only fall back to "
        "room-shape matching if no distinctive feature is visible.\n\n"
        "TASK: Find the bounding box of the annotated region in the RIGHT panel.\n\n"
        "IMPORTANT: cx and cy are pixel coordinates within the RIGHT panel ONLY — "
        f"col 0 is the LEFT edge of the RIGHT panel, col {right_w-1} is its RIGHT "
        "edge. Do NOT use coordinates from the full composite image.\n\n"
        "Return ONLY valid JSON (no markdown, no commentary).\n"
        "Put the numeric fields FIRST so you commit to coordinates before writing reasoning:\n"
        "{\n"
        f'  "cx": <center column in RIGHT panel, integer 0..{right_w-1}>,\n'
        f'  "cy": <center row in RIGHT panel, integer 0..{right_h-1}>,\n'
        f'  "w": <estimated width of annotated region in RIGHT panel, integer 1..{right_w}>,\n'
        f'  "h": <estimated height of annotated region in RIGHT panel, integer 1..{right_h}>,\n'
        '  "reasoning": "<which feature you matched and where it appears in RIGHT>"\n'
        "}"
    )

    # gemini-2.5-pro consumes many tokens for thinking before emitting JSON;
    # 512 is insufficient — use 4096 for pro, 512 for others.
    max_tok = 4096 if "pro" in model else 512
    payload = {
        "model": model, "max_tokens": max_tok,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload, timeout=90,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    # Some reasoning models (o4-mini, gpt-5, o3) return content=null with text
    # in reasoning_content or inside a content array instead of a plain string.
    raw_content = msg.get("content")
    if raw_content is None:
        raw_content = (msg.get("reasoning_content") or
                       msg.get("reasoning") or "")
    if isinstance(raw_content, list):
        # content blocks format: [{type:text, text:...}, ...]
        raw_content = " ".join(
            b.get("text", "") for b in raw_content if b.get("type") == "text"
        )
    raw = str(raw_content).strip()
    if not raw:
        raise ValueError(f"Empty response from model {model}: {resp.json()}")
    # Strip markdown fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()

    def _extract(t):
        # 1. Try direct parse
        try:
            return _j.loads(t)
        except Exception:
            pass
        # 2. Models like gemini-2.5-pro emit thinking tokens before the JSON —
        #    find the outermost {...} block and parse that.
        start = t.find('{')
        end   = t.rfind('}')
        if start != -1 and end > start:
            try:
                return _j.loads(t[start:end + 1])
            except Exception:
                pass
        # 3. Regex fallback for truncated / malformed JSON
        m = lambda k: re.search(rf'"{k}"\s*:\s*(-?\d+)', t)
        if m("cx") and m("cy"):
            return {"cx": int(m("cx").group(1)), "cy": int(m("cy").group(1)),
                    "w":  int(m("w").group(1))  if m("w")  else right_w // 4,
                    "h":  int(m("h").group(1))  if m("h")  else right_h // 4,
                    "reasoning": "(truncated)"}
        raise ValueError(f"Cannot parse JSON from response: {t[:200]!r}")

    data = _extract(raw)
    cx_raw = int(data["cx"])
    if cx_raw > right_w:
        cx_raw -= (left_w + 8)
    cx = int(np.clip(cx_raw, 0, right_w - 1))
    cy = int(np.clip(int(data["cy"]), 0, right_h - 1))
    return {"cx": cx, "cy": cy,
            "w": int(data.get("w", right_w // 4)),
            "h": int(data.get("h", right_h // 4)),
            "reasoning": str(data.get("reasoning", ""))}


def register_openrouter_vision(lidar_render, lidar_xz_bbox, ply_render, ply_world_bbox,
                               xz_polygon, model, xz_polygons=None,
                               lidar_render_display=None, n_retries=2):
    """
    Register using an OpenRouter vision model.
    Mirrors the claude_haiku_nochamfer path: build same composite, call model,
    use returned bbox center as translation seed (no Chamfer refinement),
    best-of-(1+n_retries) by meanDist to PLY wall edges.
    Returns (transform_fn, debug_img, note, {lidar_vs_result, meanDist}).
    """
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox
    if xz_polygons is None:
        xz_polygons = [xz_polygon]
    _li_disp = lidar_render_display if lidar_render_display is not None else lidar_render

    def _xz_to_li_px(xz):
        return np.stack([(xz[:, 0] - lx0) / (lx1 - lx0) * lW,
                         (xz[:, 1] - lz0) / (lz1 - lz0) * lH], axis=1)

    yellow_li_px     = _xz_to_li_px(xz_polygon)
    all_yellow_li_px = [_xz_to_li_px(p) for p in xz_polygons]

    ann_mask = np.zeros((lH, lW), np.uint8)
    for py_ in all_yellow_li_px:
        cv2.fillPoly(ann_mask, [py_.astype(np.int32)], 255)
    ann_frac = max(0.02, min(0.95, float(ann_mask.sum() / 255) / float(lH * lW)))

    max_dim = 2048
    l_scale = min(max_dim / lW, max_dim / lH, 1.0)
    p_scale = min(max_dim / pW, max_dim / pH, 1.0)
    lsW, lsH = int(lW * l_scale), int(lH * l_scale)
    psW, psH = int(pW * p_scale), int(pH * p_scale)

    l_small = cv2.resize(_li_disp, (lsW, lsH))
    for i, py_ in enumerate([p * l_scale for p in all_yellow_li_px]):
        cv2.polylines(l_small, [py_.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 255) if i == 0 else (255, 255, 0), 4)

    p_bgr = cv2.resize(ply_render, (psW, psH))
    p_lab = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    p_lab[:, :, 0] = clahe.apply(p_lab[:, :, 0])
    p_small = cv2.cvtColor(p_lab, cv2.COLOR_LAB2BGR)

    comp_h = max(lsH, psH)
    def _pad(img, h):
        if img.shape[0] >= h:
            return img
        return np.vstack([img, np.zeros((h - img.shape[0], img.shape[1], 3), np.uint8)])
    composite = np.hstack([_pad(l_small, comp_h),
                           np.full((comp_h, 8, 3), 120, np.uint8),
                           _pad(p_small, comp_h)])

    # PLY edge distance transform for meanDist scoring
    ply_gray  = cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY)
    ply_edges = cv2.Canny(ply_gray, 30, 120)
    ply_edges = cv2.dilate(ply_edges, np.ones((3, 3), np.uint8))
    dt        = cv2.distanceTransform(255 - ply_edges, cv2.DIST_L2, 5).astype(np.float32)
    PENALTY   = float(dt.max()) * 2.0

    all_pts  = np.vstack([auto_snip_lidar._densify_polygon(_xz_to_li_px(p), step=3.0)
                          for p in xz_polygons])
    homog    = np.column_stack([all_pts, np.ones(len(all_pts))])

    def cost_fn(M3):
        pp  = (M3 @ homog.T).T[:, :2]
        xs  = np.round(pp[:, 0]).astype(np.int32)
        ys  = np.round(pp[:, 1]).astype(np.int32)
        inb = (xs >= 0) & (xs < pW) & (ys >= 0) & (ys < pH)
        if inb.sum() < 0.3 * len(pp):
            return PENALTY
        vals = np.full(len(pp), PENALTY, np.float32)
        vals[inb] = dt[ys[inb], xs[inb]]
        return float(vals.mean())

    _, ang_li, _, _ = auto_snip_lidar._pca_footprint(lidar_render)
    _, ang_pl, _, _ = auto_snip_lidar._pca_footprint(ply_render)
    rot0 = ang_pl - ang_li
    src_li = np.array([float(yellow_li_px[:, 0].mean()),
                       float(yellow_li_px[:, 1].mean())])

    def _make_M3(rot, src_c, dst_c):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s, dst_c[0] - (c * src_c[0] - s * src_c[1])],
                         [s,  c, dst_c[1] - (s * src_c[0] + c * src_c[1])],
                         [0,  0, 1.0]], np.float64)

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    model_short = model.split("/")[-1]
    print(f"  [OpenRouter:{model_short}] "
          f"Calling (ann_frac={ann_frac:.0%}, n_polys={len(xz_polygons)}) ...")

    def _attempt():
        r   = _call_openrouter(composite, lsW, psW, psH, ann_frac, len(xz_polygons), model)
        dst = np.array([r["cx"] / p_scale, r["cy"] / p_scale])
        # Pick 180° flip by Chamfer cost
        best_rc, best_c = rot0, float("inf")
        for rc in [rot0, rot0 + 180]:
            Mt = _make_M3(rc, src_li, dst)
            c  = cost_fn(Mt)
            if c < best_c:
                best_c = c; best_rc = rc
        return _make_M3(best_rc, src_li, dst), best_c, best_rc, r["reasoning"]

    best_M3, best_cost, best_rot, best_reasoning = _attempt()
    print(f"  [OpenRouter:{model_short}] cx={src_li[0]:.0f} meanDist={best_cost:.2f}px")
    for retry in range(n_retries):
        try:
            M, c, rot, reas = _attempt()
            print(f"  [OpenRouter:{model_short}] retry {retry+1}: meanDist={c:.2f}px")
            if c < best_cost:
                best_M3, best_cost, best_rot, best_reasoning = M, c, rot, reas
        except Exception as e:
            print(f"  [OpenRouter:{model_short}] retry {retry+1} failed: {e}")

    note = (f"OpenRouter[{model_short}] rot={best_rot:.1f}° "
            f"meanDist={best_cost:.2f}px")
    print(f"  {note}")

    all_ply_px_r = [_apply(best_M3, p) for p in all_yellow_li_px]

    def transform_fn(xz_pts):
        pp = _apply(best_M3, _xz_to_li_px(xz_pts))
        world = np.empty((len(pp), 2))
        world[:, 0] = px0 + pp[:, 0] * (px1 - px0) / pW
        world[:, 1] = py1 - pp[:, 1] * (py1 - py0) / pH
        return world

    debug_img = ply_render.copy()
    for i, apx in enumerate(all_ply_px_r):
        cv2.polylines(debug_img, [apx.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0) if i == 0 else (255, 255, 0), 3)

    li_panel = _li_disp.copy()
    for i, lpy in enumerate(all_yellow_li_px):
        cv2.polylines(li_panel, [lpy.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 255) if i == 0 else (0, 255, 128), 3)
    pl_panel = debug_img.copy()
    if li_panel.shape[0] != pl_panel.shape[0]:
        tH = pl_panel.shape[0]
        li_panel = cv2.resize(li_panel, (int(li_panel.shape[1] * tH / li_panel.shape[0]), tH))
    cv2.putText(pl_panel, note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),   4)
    cv2.putText(pl_panel, note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    lvr = np.hstack([li_panel, pl_panel])

    return transform_fn, debug_img, note, {"lidar_vs_result": lvr, "meanDist": best_cost}


# ── Few-shot example storage (populated when site 20002 is processed) ──────────
_EXAMPLE = {
    "composite": None,  # full composite BGR image
    "left_w":    None,
    "right_w":   None,
    "right_h":   None,
    "gt_cx":     None,  # GT centroid in right-panel pixel coords
    "gt_cy":     None,
    "gt_w":      None,
    "gt_h":      None,
}


def _store_example(composite, left_w, right_w, right_h, p_scale,
                   gt_render_polys, render_shape):
    """Rasterise GT polys and save right-panel-scaled centroid for few-shot."""
    H, W = render_shape[:2]
    gt_mask = np.zeros((H, W), np.uint8)
    for p in gt_render_polys:
        pts = np.clip(p.astype(np.int32), [[0, 0]], [[W - 1, H - 1]])
        cv2.fillPoly(gt_mask, [pts], 255)
    gt_ys, gt_xs = np.where(gt_mask > 0)
    if len(gt_xs) == 0:
        print("  [example] WARNING: no GT pixels found, example not stored")
        return
    _EXAMPLE["composite"] = composite.copy()
    _EXAMPLE["left_w"]    = left_w
    _EXAMPLE["right_w"]   = right_w
    _EXAMPLE["right_h"]   = right_h
    _EXAMPLE["gt_cx"] = int(round(float(gt_xs.mean()) * p_scale))
    _EXAMPLE["gt_cy"] = int(round(float(gt_ys.mean()) * p_scale))
    _EXAMPLE["gt_w"]  = int(round(float(gt_xs.max() - gt_xs.min()) * p_scale))
    _EXAMPLE["gt_h"]  = int(round(float(gt_ys.max() - gt_ys.min()) * p_scale))
    print(f"  [example] Stored {EXAMPLE_SITE_ID}: "
          f"cx={_EXAMPLE['gt_cx']} cy={_EXAMPLE['gt_cy']} "
          f"w={_EXAMPLE['gt_w']} h={_EXAMPLE['gt_h']}")


def _call_openrouter_fewshot(composite_bgr, left_w, right_w, right_h,
                             ann_frac, n_polys, model):
    """
    Like _call_openrouter but prepends the 20002 example with its correct answer.
    Sends TWO images in one user message: example composite + answer, then test composite.
    """
    import re, json as _j

    if _EXAMPLE["composite"] is None:
        raise RuntimeError("Example not stored — process site 20002 first")

    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_KEY not set")

    def _enc(img):
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return base64.standard_b64encode(buf.tobytes()).decode()

    ex_b64 = _enc(_EXAMPLE["composite"])
    te_b64 = _enc(composite_bgr)

    ex_lw = _EXAMPLE["left_w"]
    ex_rw = _EXAMPLE["right_w"]
    ex_rh = _EXAMPLE["right_h"]
    ex_cx = _EXAMPLE["gt_cx"]
    ex_cy = _EXAMPLE["gt_cy"]
    ex_w  = _EXAMPLE["gt_w"]
    ex_h  = _EXAMPLE["gt_h"]

    poly_desc = (f"{n_polys} separate annotated regions" if n_polys > 1
                 else "one annotated region")

    example_text = (
        "EXAMPLE — study this so you understand the task:\n\n"
        "The image BELOW is a side-by-side composite:\n"
        f"  LEFT ({ex_lw}px wide): iPhone LiDAR scan from INSIDE an excavation trench. "
        "Bright cyan outline = yellow spray-paint annotation on the trench floor/walls.\n"
        f"  RIGHT ({ex_rw}×{ex_rh}px): Top-down photogrammetry (aerial/drone view) of the "
        "SAME site looking straight down. No annotation is shown.\n\n"
        "KEY INSIGHT: The LEFT shows wall FACES from inside; the RIGHT shows wall TOPS "
        "from above. They look different but show the same physical space.\n\n"
        f"CORRECT ANSWER for this example:\n"
        f'{{"cx": {ex_cx}, "cy": {ex_cy}, "w": {ex_w}, "h": {ex_h}}}\n'
        "(cx and cy are the center of the annotated region in the RIGHT panel only.)"
    )

    task_text = (
        "YOUR TASK (a DIFFERENT archaeological site — not the example):\n\n"
        "The image below is another side-by-side composite with the same format:\n"
        f"  LEFT ({left_w}px wide): LiDAR scan with {poly_desc} in cyan.\n"
        f"  RIGHT ({right_w}×{right_h}px): Top-down photogrammetry of that site.\n\n"
        f"The annotation fills about {ann_frac:.0%} of the LEFT scan.\n"
        "CRITICAL: The RIGHT panel shows the ENTIRE site — the matching region "
        "may appear much smaller than in LEFT and can be ANYWHERE in the panel.\n"
        "Scale is NOT guaranteed to match (the PLY may cover 2–10× more area).\n\n"
        "MATCHING STRATEGY: First identify distinctive internal features in the annotated "
        "area (stone blocks, pillars, hearths, wall junctions). Find those same features "
        "in the aerial RIGHT panel. Use the example as a guide for how features look from "
        "above vs from the side.\n\n"
        "Return ONLY valid JSON, numeric fields FIRST:\n"
        "{\n"
        f'  "cx": <center col in RIGHT panel, integer 0..{right_w-1}>,\n'
        f'  "cy": <center row in RIGHT panel, integer 0..{right_h-1}>,\n'
        f'  "w": <estimated width in RIGHT panel, integer 1..{right_w}>,\n'
        f'  "h": <estimated height in RIGHT panel, integer 1..{right_h}>,\n'
        '  "reasoning": "<which feature matched and where in RIGHT>"\n'
        "}"
    )

    max_tok = 4096 if "pro" in model else 1024
    payload = {
        "model": model,
        "max_tokens": max_tok,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": example_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ex_b64}"}},
            {"type": "text",      "text": task_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{te_b64}"}},
        ]}],
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=120,
    )
    resp.raise_for_status()
    msg2 = resp.json()["choices"][0]["message"]
    raw_content2 = msg2.get("content")
    if raw_content2 is None:
        raw_content2 = (msg2.get("reasoning_content") or msg2.get("reasoning") or "")
    if isinstance(raw_content2, list):
        raw_content2 = " ".join(
            b.get("text", "") for b in raw_content2 if b.get("type") == "text"
        )
    raw = str(raw_content2).strip()
    if not raw:
        raise ValueError(f"Empty response from model {model}: {resp.json()}")
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()

    def _extract(t):
        import re as _re, json as _j2
        try:
            return _j2.loads(t)
        except Exception:
            pass
        start = t.find('{'); end = t.rfind('}')
        if start != -1 and end > start:
            try:
                return _j2.loads(t[start:end + 1])
            except Exception:
                pass
        m = lambda k: _re.search(rf'"{k}"\s*:\s*(-?\d+)', t)
        if m("cx") and m("cy"):
            return {"cx": int(m("cx").group(1)), "cy": int(m("cy").group(1)),
                    "w": int(m("w").group(1)) if m("w") else right_w // 4,
                    "h": int(m("h").group(1)) if m("h") else right_h // 4,
                    "reasoning": "(truncated)"}
        raise ValueError(f"Cannot parse JSON from response: {t[:200]!r}")

    data = _extract(raw)
    cx_raw = int(data["cx"])
    if cx_raw > right_w:
        cx_raw -= (left_w + 8)
    cx = int(np.clip(cx_raw, 0, right_w - 1))
    cy = int(np.clip(int(data["cy"]), 0, right_h - 1))
    return {"cx": cx, "cy": cy,
            "w": int(data.get("w", right_w // 4)),
            "h": int(data.get("h", right_h // 4)),
            "reasoning": str(data.get("reasoning", ""))}


def register_openrouter_fewshot(lidar_render, lidar_xz_bbox, ply_render, ply_world_bbox,
                                xz_polygon, model, xz_polygons=None,
                                lidar_render_display=None, n_retries=2):
    """
    Like register_openrouter_vision but uses a few-shot prompt with site 20002 as example.
    Only meaningful for sites other than 20002 (the example site).
    """
    lH, lW = lidar_render.shape[:2]
    pH, pW = ply_render.shape[:2]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    px0, py0, px1, py1 = ply_world_bbox
    if xz_polygons is None:
        xz_polygons = [xz_polygon]
    _li_disp = lidar_render_display if lidar_render_display is not None else lidar_render

    def _xz_to_li_px(xz):
        return np.stack([(xz[:, 0] - lx0) / (lx1 - lx0) * lW,
                         (xz[:, 1] - lz0) / (lz1 - lz0) * lH], axis=1)

    yellow_li_px     = _xz_to_li_px(xz_polygon)
    all_yellow_li_px = [_xz_to_li_px(p) for p in xz_polygons]

    ann_mask = np.zeros((lH, lW), np.uint8)
    for py_ in all_yellow_li_px:
        cv2.fillPoly(ann_mask, [py_.astype(np.int32)], 255)
    ann_frac = max(0.02, min(0.95, float(ann_mask.sum() / 255) / float(lH * lW)))

    max_dim = 2048
    l_scale = min(max_dim / lW, max_dim / lH, 1.0)
    p_scale = min(max_dim / pW, max_dim / pH, 1.0)
    lsW, lsH = int(lW * l_scale), int(lH * l_scale)
    psW, psH = int(pW * p_scale), int(pH * p_scale)

    l_small = cv2.resize(_li_disp, (lsW, lsH))
    for i, py_ in enumerate([p * l_scale for p in all_yellow_li_px]):
        cv2.polylines(l_small, [py_.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 255) if i == 0 else (255, 255, 0), 4)

    p_bgr = cv2.resize(ply_render, (psW, psH))
    p_lab = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    p_lab[:, :, 0] = clahe.apply(p_lab[:, :, 0])
    p_small = cv2.cvtColor(p_lab, cv2.COLOR_LAB2BGR)

    comp_h = max(lsH, psH)
    def _pad(img, h):
        if img.shape[0] >= h:
            return img
        return np.vstack([img, np.zeros((h - img.shape[0], img.shape[1], 3), np.uint8)])
    composite = np.hstack([_pad(l_small, comp_h),
                           np.full((comp_h, 8, 3), 120, np.uint8),
                           _pad(p_small, comp_h)])

    ply_gray  = cv2.cvtColor(ply_render, cv2.COLOR_BGR2GRAY)
    ply_edges = cv2.Canny(ply_gray, 30, 120)
    ply_edges = cv2.dilate(ply_edges, np.ones((3, 3), np.uint8))
    dt        = cv2.distanceTransform(255 - ply_edges, cv2.DIST_L2, 5).astype(np.float32)
    PENALTY   = float(dt.max()) * 2.0

    all_pts = np.vstack([auto_snip_lidar._densify_polygon(_xz_to_li_px(p), step=3.0)
                         for p in xz_polygons])
    homog   = np.column_stack([all_pts, np.ones(len(all_pts))])

    def cost_fn(M3):
        pp  = (M3 @ homog.T).T[:, :2]
        xs  = np.round(pp[:, 0]).astype(np.int32)
        ys  = np.round(pp[:, 1]).astype(np.int32)
        inb = (xs >= 0) & (xs < pW) & (ys >= 0) & (ys < pH)
        if inb.sum() < 0.3 * len(pp):
            return PENALTY
        vals = np.full(len(pp), PENALTY, np.float32)
        vals[inb] = dt[ys[inb], xs[inb]]
        return float(vals.mean())

    _, ang_li, _, _ = auto_snip_lidar._pca_footprint(lidar_render)
    _, ang_pl, _, _ = auto_snip_lidar._pca_footprint(ply_render)
    rot0 = ang_pl - ang_li
    src_li = np.array([float(yellow_li_px[:, 0].mean()),
                       float(yellow_li_px[:, 1].mean())])

    def _make_M3(rot, src_c, dst_c):
        r = np.radians(rot); c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s, dst_c[0] - (c * src_c[0] - s * src_c[1])],
                         [s,  c, dst_c[1] - (s * src_c[0] + c * src_c[1])],
                         [0,  0, 1.0]], np.float64)

    def _apply(M3, pts):
        h = np.column_stack([pts, np.ones(len(pts))])
        return (M3 @ h.T).T[:, :2]

    model_short = model.split("/")[-1]
    tag = f"[OR-fewshot:{model_short}]"
    print(f"  {tag} Calling (ann_frac={ann_frac:.0%}, n_polys={len(xz_polygons)}) ...")

    def _attempt():
        r   = _call_openrouter_fewshot(composite, lsW, psW, psH,
                                       ann_frac, len(xz_polygons), model)
        dst = np.array([r["cx"] / p_scale, r["cy"] / p_scale])
        best_rc, best_c = rot0, float("inf")
        for rc in [rot0, rot0 + 180]:
            Mt = _make_M3(rc, src_li, dst)
            c  = cost_fn(Mt)
            if c < best_c:
                best_c = c; best_rc = rc
        return _make_M3(best_rc, src_li, dst), best_c, best_rc, r["reasoning"]

    best_M3, best_cost, best_rot, best_reasoning = _attempt()
    print(f"  {tag} meanDist={best_cost:.2f}px  rot={best_rot:.1f}°")
    for retry in range(n_retries):
        try:
            M, c, rot, reas = _attempt()
            print(f"  {tag} retry {retry+1}: meanDist={c:.2f}px")
            if c < best_cost:
                best_M3, best_cost, best_rot, best_reasoning = M, c, rot, reas
        except Exception as e:
            print(f"  {tag} retry {retry+1} failed: {e}")

    note = (f"OR-fewshot[{model_short}] rot={best_rot:.1f}° meanDist={best_cost:.2f}px")
    print(f"  {note}")

    all_ply_px_r = [_apply(best_M3, p) for p in all_yellow_li_px]

    def transform_fn(xz_pts):
        pp = _apply(best_M3, _xz_to_li_px(xz_pts))
        world = np.empty((len(pp), 2))
        world[:, 0] = px0 + pp[:, 0] * (px1 - px0) / pW
        world[:, 1] = py1 - pp[:, 1] * (py1 - py0) / pH
        return world

    debug_img = ply_render.copy()
    for i, apx in enumerate(all_ply_px_r):
        cv2.polylines(debug_img, [apx.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 0) if i == 0 else (255, 255, 0), 3)

    li_panel = _li_disp.copy()
    for i, lpy in enumerate(all_yellow_li_px):
        cv2.polylines(li_panel, [lpy.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 255) if i == 0 else (0, 255, 128), 3)
    pl_panel = debug_img.copy()
    if li_panel.shape[0] != pl_panel.shape[0]:
        tH = pl_panel.shape[0]
        li_panel = cv2.resize(li_panel, (int(li_panel.shape[1] * tH / li_panel.shape[0]), tH))
    cv2.putText(pl_panel, note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),   4)
    cv2.putText(pl_panel, note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    lvr = np.hstack([li_panel, pl_panel])

    return transform_fn, debug_img, note, {"lidar_vs_result": lvr, "meanDist": best_cost}


# ── Main evaluation ───────────────────────────────────────────────────────────

# Methods that take the standard 5-arg signature:
#   (lidar_render, lidar_xz_bbox, ply_render, ply_world_bbox, xz_polygon)
STANDARD_METHODS = [
    ("rgb_pca",      auto_snip_lidar.register_lidar_to_ply_world),
    ("pca_chamfer",  auto_snip_lidar.register_lidar_to_ply_world_pca_chamfer),
    ("prerot_akaze", auto_snip_lidar.register_lidar_to_ply_world_prerot_akaze),
    ("phase_corr",   auto_snip_lidar.register_lidar_to_ply_world_phase_corr),
    ("annot_bndry",  auto_snip_lidar.register_lidar_to_ply_world_annot_boundary),
    ("dist_pca",     auto_snip_lidar.register_lidar_to_ply_world_dist_pca),
    ("mutual_info",  auto_snip_lidar.register_lidar_to_ply_world_mutual_info),
    ("edges",        auto_snip_lidar.register_lidar_to_ply_world_edges),
]

# Zero-shot OR models (all 4 sites)
OR_ZEROSHOT_MODELS = [
    # Previously run — skipped when --new-only is passed
    ("gemini_25pro",     "google/gemini-2.5-pro"),
    ("gemini_25flash",   "google/gemini-2.5-flash"),
    ("gpt4o",            "openai/gpt-4o"),
    # New models
    ("qwen3vl_235b",     "qwen/qwen3-vl-235b-a22b-instruct"),  # best spatial grounding
    ("qwen3vl_32b",      "qwen/qwen3-vl-32b-instruct"),        # smaller/faster Qwen3-VL
    ("o4mini",           "openai/o4-mini"),                     # reasoning + vision
    ("llama4_maverick",  "meta-llama/llama-4-maverick"),        # cheap, 1M ctx
    ("gpt5",             "openai/gpt-5"),                       # latest OpenAI vision
]
EXISTING_OR_NAMES = {"gemini_25pro", "gemini_25flash", "gpt4o"}

# Few-shot OR models (example=20002; test sites = 20003/20005/21001)
OR_FEWSHOT_MODELS = [
    ("gemini25flash_fs", "google/gemini-2.5-flash"),
    ("qwen3vl_235b_fs",  "qwen/qwen3-vl-235b-a22b-instruct"),
    ("o4mini_fs",        "openai/o4-mini"),
    ("gpt5_fs",          "openai/gpt-5"),
]

ALL_METHOD_NAMES = (
    [m[0] for m in STANDARD_METHODS]
    + ["claude_haiku_nochamfer", "claude_haiku_chamfer"]
    + [m[0] for m in OR_ZEROSHOT_MODELS]
    + [m[0] for m in OR_FEWSHOT_MODELS]
)

results_table = []  # list of (method, site_id, centroid_err_m, iou)


def _run_and_record(mname, site_id, result_or_exc,
                    xz_polygons, render_img, render_world_bbox,
                    lidar_disp, lidar_polys_px, gt_render_polys):
    """Compute metrics, save debug image, append to results_table."""
    if isinstance(result_or_exc, Exception):
        print(f"  [{mname}] FAILED: {result_or_exc}")
        results_table.append((mname, site_id, float("inf"), 0.0))
        return

    result = result_or_exc
    transform_fn = result[0]

    placed_world  = [transform_fn(p) for p in xz_polygons]
    placed_render = [world_to_render_px(pw, render_world_bbox, render_img.shape)
                     for pw in placed_world]

    metrics = compute_metrics(gt_render_polys, placed_render, render_img.shape)
    err_m   = metrics["centroid_err_m"]
    iou     = metrics["iou"]
    note    = result[2] if len(result) > 2 else mname
    print(f"  [{mname}] centroid_err={err_m:.3f}m  IoU={iou:.3f}  ({note})")

    results_table.append((mname, site_id, err_m, iou))

    label = f"{mname}: err={err_m:.2f}m IoU={iou:.2f}"
    dbg   = make_eval_debug(lidar_disp, lidar_polys_px, render_img,
                            gt_render_polys, placed_render, label)
    out   = os.path.join(EVAL_DIR, f"{site_id}_{mname}.png")
    cv2.imwrite(out, dbg)
    print(f"  Saved: {out}")


for site in SITES:
    site_id = site["id"]
    print(f"\n{'='*60}")
    print(f"  Site {site_id}")
    print(f"{'='*60}")

    with open(site["json_file"]) as f:
        job_data = json.load(f)
    job       = job_data[0]
    top_job   = job["top"]
    usdz_path = job["annotations"][0]

    top_id = find_mesh_by_pgram_job(top_job)
    if top_id is None:
        print(f"  ERROR: no PLY for job {top_job}, skipping")
        continue

    # Load PLY top cloud and render
    print(f"  Loading PLY (top={top_id}) ...")
    try:
        top_bin   = find_top_bin(top_id)
        top_cloud = cc.loadPointCloud(top_bin)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        continue
    print(f"  Top cloud: {top_cloud.size()} pts")
    render_img, render_world_bbox = render_topdown(top_cloud)
    rH, rW = render_img.shape[:2]

    # Save PLY render for reference
    cv2.imwrite(os.path.join(EVAL_DIR, f"{site_id}_ply_render.png"), render_img)

    # Process USDZ
    print(f"  Processing USDZ: {usdz_path}")
    try:
        lidar = auto_snip_lidar.process_usdz(usdz_path)
    except Exception as e:
        print(f"  ERROR processing USDZ: {e}")
        continue

    lidar_render  = lidar["lidar_render"]
    lidar_disp    = lidar.get("lidar_render_display", lidar_render)
    lidar_xz_bbox = lidar["lidar_xz_bbox"]
    xz_polygon    = lidar["xz_polygon"]
    xz_polygons   = lidar.get("xz_polygons") or [xz_polygon]
    lx0, lz0, lx1, lz1 = lidar_xz_bbox
    lH, lW = lidar_render.shape[:2]

    def _xz_to_li_px(xz):
        return np.stack([(xz[:, 0] - lx0) / (lx1 - lx0) * lW,
                         (xz[:, 1] - lz0) / (lz1 - lz0) * lH], axis=1)
    lidar_polys_px = [_xz_to_li_px(p) for p in xz_polygons]

    # Extract GT polygons
    print(f"  Extracting GT from {site['ortho_ann']} ...")
    raw_img = cv2.imread(site["ortho_raw"])
    try:
        gt_ortho_polys = extract_gt_polygons(site["ortho_raw"], site["ortho_ann"])
    except Exception as e:
        print(f"  ERROR extracting GT: {e}")
        continue
    if not gt_ortho_polys:
        print("  ERROR: no GT polygons found in diff, skipping site")
        continue
    gt_render_polys = map_gt_to_render(gt_ortho_polys, raw_img,
                                       render_world_bbox, render_img.shape)

    # Save GT overlay
    gt_debug = render_img.copy()
    for p in gt_render_polys:
        cv2.polylines(gt_debug, [p.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(EVAL_DIR, f"{site_id}_gt.png"), gt_debug)

    _args = (lidar_render, lidar_xz_bbox, render_img, render_world_bbox, xz_polygon)

    # Build composite for this site (used both for fewshot storage and running OR methods)
    _lH, _lW = lidar_render.shape[:2]
    _pH, _pW = render_img.shape[:2]
    _max_dim = 2048
    _l_scale = min(_max_dim / _lW, _max_dim / _lH, 1.0)
    _p_scale = min(_max_dim / _pW, _max_dim / _pH, 1.0)
    _lsW, _lsH = int(_lW * _l_scale), int(_lH * _l_scale)
    _psW, _psH = int(_pW * _p_scale), int(_pH * _p_scale)
    _l_small = cv2.resize(lidar_disp, (_lsW, _lsH))
    lx0_, lz0_, lx1_, lz1_ = lidar_xz_bbox
    for _i, _py in enumerate([_xz_to_li_px(p) * _l_scale for p in xz_polygons]):
        cv2.polylines(_l_small, [_py.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 255, 255) if _i == 0 else (255, 255, 0), 4)
    _p_bgr = cv2.resize(render_img, (_psW, _psH))
    _p_lab = cv2.cvtColor(_p_bgr, cv2.COLOR_BGR2LAB)
    _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    _p_lab[:, :, 0] = _clahe.apply(_p_lab[:, :, 0])
    _p_small = cv2.cvtColor(_p_lab, cv2.COLOR_LAB2BGR)
    _comp_h = max(_lsH, _psH)
    def _pad_c(img, h):
        if img.shape[0] >= h: return img
        return np.vstack([img, np.zeros((h - img.shape[0], img.shape[1], 3), np.uint8)])
    _site_composite = np.hstack([_pad_c(_l_small, _comp_h),
                                  np.full((_comp_h, 8, 3), 120, np.uint8),
                                  _pad_c(_p_small, _comp_h)])

    # Store site 20002 as few-shot example
    if site_id == EXAMPLE_SITE_ID:
        _store_example(_site_composite, _lsW, _psW, _psH, _p_scale,
                       gt_render_polys, render_img.shape)

    if not OR_ONLY:
        # Standard 5-arg methods
        for mname, mfn in STANDARD_METHODS:
            print(f"\n  [{mname}] ...")
            try:
                res = mfn(*_args)
            except Exception as e:
                res = e
            _run_and_record(mname, site_id, res,
                            xz_polygons, render_img, render_world_bbox,
                            lidar_disp, lidar_polys_px, gt_render_polys)

        # Claude Vision methods
        if os.environ.get("ANTHROPIC_API_KEY"):
            for cv_name, use_chamfer in [("claude_haiku_nochamfer", False),
                                         ("claude_haiku_chamfer",   True)]:
                print(f"\n  [{cv_name}] ...")
                try:
                    res = auto_snip_lidar.register_lidar_to_ply_world_claude_vision(
                        *_args,
                        xz_polygons=xz_polygons,
                        model="claude-haiku-4-5-20251001",
                        use_chamfer=use_chamfer,
                        lidar_render_display=lidar_disp,
                    )
                except Exception as e:
                    res = e
                _run_and_record(cv_name, site_id, res,
                                xz_polygons, render_img, render_world_bbox,
                                lidar_disp, lidar_polys_px, gt_render_polys)
        else:
            print("  ANTHROPIC_API_KEY not set — skipping Claude Vision methods")

    # OpenRouter zero-shot methods
    if os.environ.get("OPEN_ROUTER_KEY"):
        zs_models = OR_ZEROSHOT_MODELS
        if PRO_ONLY:
            zs_models = [("gemini_25pro", "google/gemini-2.5-pro")]
        elif NEW_ONLY:
            zs_models = [(n, m) for n, m in OR_ZEROSHOT_MODELS
                         if n not in EXISTING_OR_NAMES]
        for or_name, or_model in zs_models:
            print(f"\n  [{or_name}] ...")
            try:
                res = register_openrouter_vision(
                    lidar_render, lidar_xz_bbox, render_img, render_world_bbox,
                    xz_polygon, model=or_model,
                    xz_polygons=xz_polygons,
                    lidar_render_display=lidar_disp,
                )
            except Exception as e:
                res = e
            _run_and_record(or_name, site_id, res,
                            xz_polygons, render_img, render_world_bbox,
                            lidar_disp, lidar_polys_px, gt_render_polys)

        # Few-shot methods — skip the example site (20002) itself
        if site_id != EXAMPLE_SITE_ID and _EXAMPLE["composite"] is not None and not PRO_ONLY:
            for fs_name, fs_model in OR_FEWSHOT_MODELS:
                if NEW_ONLY and fs_name.replace("_fs", "") in EXISTING_OR_NAMES:
                    # still run fewshot for existing models (it's new data)
                    pass
                print(f"\n  [{fs_name}] ...")
                try:
                    res = register_openrouter_fewshot(
                        lidar_render, lidar_xz_bbox, render_img, render_world_bbox,
                        xz_polygon, model=fs_model,
                        xz_polygons=xz_polygons,
                        lidar_render_display=lidar_disp,
                    )
                except Exception as e:
                    res = e
                _run_and_record(fs_name, site_id, res,
                                xz_polygons, render_img, render_world_bbox,
                                lidar_disp, lidar_polys_px, gt_render_polys)
    else:
        print("  OPEN_ROUTER_KEY not set — skipping OpenRouter methods")


# ── Ranking table ─────────────────────────────────────────────────────────────
# Build lookup: (method, site_id) → (centroid_err_m, iou)
lookup = {(m, s): (e, iou) for m, s, e, iou in results_table}
site_ids = [s["id"] for s in SITES]

header = (f"{'Method':<28} | " +
          " | ".join(f"{sid:>8}" for sid in site_ids) +
          f" | {'Mean':>8}")
sep = "-" * len(header)

print(f"\n\n{'='*len(header)}")
print("RANKING TABLE  (centroid error in metres, lower is better)")
print(f"{'='*len(header)}")
print(header)
print(sep)

method_means = []
for mname in ALL_METHOD_NAMES:
    vals = []
    cols = [f"{mname:<28}"]
    for sid in site_ids:
        key = (mname, sid)
        if key in lookup:
            e, _ = lookup[key]
            s = f"{e:.3f}m" if e < 1e9 else "FAIL"
            cols.append(f"{s:>8}")
            if e < 1e9:
                vals.append(e)
        else:
            cols.append(f"{'N/A':>8}")
    mean_e = float(np.mean(vals)) if vals else float("inf")
    ms = f"{mean_e:.3f}m" if mean_e < 1e9 else "FAIL"
    cols.append(f"{ms:>8}")
    print(" | ".join(cols))
    method_means.append((mname, mean_e))

print(f"\n{'='*len(header)}")
print("RANKED BY MEAN CENTROID ERROR (best → worst):")
method_means.sort(key=lambda x: x[1])
for rank, (mname, mean_e) in enumerate(method_means, 1):
    ms = f"{mean_e:.3f}m" if mean_e < 1e9 else "FAIL"
    print(f"  {rank:2}. {mname:<28}  {ms}")

# Also show IoU ranking
iou_means = []
for mname in ALL_METHOD_NAMES:
    vals = [lookup[(mname, s)][1]
            for s in site_ids if (mname, s) in lookup and lookup[(mname, s)][0] < 1e9]
    iou_means.append((mname, float(np.mean(vals)) if vals else 0.0))

print(f"\nRANKED BY MEAN IoU (best → worst):")
iou_means.sort(key=lambda x: -x[1])
for rank, (mname, miou) in enumerate(iou_means, 1):
    print(f"  {rank:2}. {mname:<28}  {miou:.3f}")

ranking_path = os.path.join(EVAL_DIR, "ranking.txt")
with open(ranking_path, "w") as fout:
    fout.write("RANKED BY MEAN CENTROID ERROR:\n")
    for rank, (mname, mean_e) in enumerate(method_means, 1):
        ms = f"{mean_e:.3f}m" if mean_e < 1e9 else "FAIL"
        fout.write(f"  {rank:2}. {mname:<28}  {ms}\n")
    fout.write("\nRANKED BY MEAN IoU:\n")
    for rank, (mname, miou) in enumerate(iou_means, 1):
        fout.write(f"  {rank:2}. {mname:<28}  {miou:.3f}\n")
print(f"\nRanking saved: {ranking_path}")
print("Done.")
