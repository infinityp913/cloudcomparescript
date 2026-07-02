import json
import re
import cloudComPy as cc
import cloudComPy.PoissonRecon
import os
import sys
import numpy as np

from pre_snip_script import save_mesh, DATA_DIR, save_project
import glob
import tarp_progress

# Windows consoles default to cp1252; keep any stray Unicode in log lines from
# crashing the run (the dashboard captures stdout, so an encode error aborts it).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

POINT_CLOUD_DIR = "Data"


def _volumetrics_dir():
    """Root of this season's Volumetrics_<year> folder. The dashboard passes it via
    TARP_VOLUMETRICS_DIR (built from base_path + season_year in config.yaml); when this
    script is run standalone, fall back to a local 'Volumetrics' folder so the run still
    succeeds."""
    return os.environ.get("TARP_VOLUMETRICS_DIR") or f"{POINT_CLOUD_DIR}/Volumetrics"


def _trench_name(su_number):
    """'Trench NNNNN' for an SU id (e.g. 20001 -> 'Trench 20000'), matching the lab's
    trench-folder convention. Returns None if the SU id has no digits."""
    m = re.search(r"\d+", str(su_number))
    if not m:
        return None
    return f"Trench {(int(m.group(0)) // 1000) * 1000}"

# The top mesh's phantom skirt (Poisson's extrapolated closing surface reaching
# past the real SU edge) is trimmed by each vertex's DISTANCE to the input top
# cloud, not by Poisson density. Interior vertices — even in sparsely-sampled
# patches — are always within ~one sample spacing of a real point, so distance-
# trimming removes only the far outer skirt and can never punch interior holes
# (which density-thresholding did). The keep/cut split is auto-detected with Otsu
# on the bimodal distance distribution (interior ≈ 0 vs skirt ≫ 0). This replaces
# the manual step of tightening the SF display range on '<su>_top_raw' in CC.
#
# Safety cap: never remove more than this % of faces (guards a bad Otsu split).
# Removing high-distance faces only ever nibbles the outer rim, never the interior,
# so erring low just leaves a little skirt. Raise if a skirt survives; lower if the
# rim gets over-trimmed.
TOP_MESH_DIST_MAX_CUT_PCT = 50

# Point cap for the TOP-mesh Poisson only. The top mesh exists solely to derive the
# SU's 2-D footprint for the SU sheet, so an oversized top cloud (e.g. a poorly-cropped
# SU) just yields a huge OBJ that later chokes the Blender/QGIS shapefile step. When the
# top cloud exceeds this, a random-subsampled COPY is fed to the top Poisson; the full
# top_cloud is left untouched for the merged volume, the 2.5D volume, and the distance
# trim reference. Normal SUs (~25-30k pts) are far below this and pass through unchanged.
TOP_MESH_MAX_POINTS = 200_000

# Octree depth for the merged VOLUME Poisson reconstruction. Thin SUs (a deposit
# only ~1 cm thick over surfaces with tens of cm of relief) pinch to just a few
# octree cells; at depth 11 (~1 mm cells) Poisson's smoothing can't hold the two
# opposing surfaces apart in the thinnest spots and leaves holes. A deeper octree
# (finer cells -> more cells across the slab) resolves the thin slab. Raise to 13
# if holes persist on very thin SUs; lower to 11 (the doc default) if memory/time
# is tight on large clouds. The top mesh stays at 11 — it's a single surface with
# no thin-slab problem.
MERGED_MESH_POISSON_DEPTH = 12



def find_combined_bin(point_cloud_dir, su):
    """
    Find the operator's combined snip bin for this SU in Data/SU<su>/.

    New manual workflow: after snipping the SU in CloudCompare, the operator
    saves BOTH cropped clouds (top + bottom) into a single project bin in that
    SU's folder. The filename is flexible — just the SU number with an optional
    'SU'/'su' prefix, any case — e.g. '20001.bin', 'SU20001.bin', 'su20001.bin'.

    Pre-snip's own '*_with_dist_*.bin' inputs and post-snip's '<su>_post_snip.bin'
    output are intentionally NOT matched. Returns the bin path (newest by mtime if
    several match) or None.
    """
    folder = os.path.join(point_cloud_dir, f"SU{su}") if su else None
    if not folder or not os.path.isdir(folder):
        print(f"  Folder not found: {folder}")
        return None

    # Accept '<su>.bin' with an optional case-insensitive 'su' prefix and nothing
    # else, so the pre-snip and post-snip bins in the same folder are excluded.
    name_re = re.compile(rf"^(su)?{re.escape(str(su))}\.bin$", re.IGNORECASE)
    matches = [
        p for p in glob.glob(os.path.join(folder, "*.bin"))
        if name_re.match(os.path.basename(p))
    ]
    if not matches:
        return None

    bin_path = max(matches, key=os.path.getmtime)
    if len(matches) > 1:
        print(f"  Multiple combined bins for SU {su}; using newest: {os.path.basename(bin_path)}")
    return bin_path


def _entity_names(entity):
    """The entity's own name plus its parent group's name (when reachable).

    cloudComPy's importFile usually flattens away the group structure, so the
    parent name is a best-effort extra signal, not something to rely on.
    """
    names = [entity.getName()]
    try:
        parent = entity.getParent()
        if parent is not None:
            names.append(parent.getName())
    except Exception:
        pass
    return [n for n in names if n]


def _pgram_of(entity):
    """First Pgram_Job_<n> number in the entity (or parent) name, as a string."""
    for name in _entity_names(entity):
        m = re.search(r"Pgram_Job_(\d+)", name)
        if m:
            return m.group(1)
    return ""


def _role_from_name(entity):
    """'top'/'bottom'/'' from a 'top'/'bottom' marker in the entity/parent name."""
    joined = " ".join(n.lower() for n in _entity_names(entity))
    has_top = "top" in joined
    has_bot = "bottom" in joined
    if has_top and not has_bot:
        return "top"
    if has_bot and not has_top:
        return "bottom"
    return ""


def load_top_bottom_from_bin(bin_path, top_pgram, bot_pgram):
    """
    Load the combined snip bin and return (top_cloud, bottom_cloud).

    The bin holds the two cropped point clouds. We identify which is which by the
    Pgram_Job_<n> number embedded in each cloud's name, matched against the
    top/bottom Pgram numbers from input.json (the robust signal, since importFile
    flattens away the 'top'/'bottom' group names). Fallbacks, in order: a
    'top'/'bottom' marker still present on the name, then plain file order.

    Returns (top_cloud, bottom_cloud), or (None, None) on failure.
    """
    try:
        imported = cc.importFile(bin_path)
    except Exception as e:
        print(f"  Failed to import {os.path.basename(bin_path)}: {e}")
        return None, None

    # importFile returns (meshes, clouds); flatten defensively and keep clouds.
    clouds = []
    for group in (imported if isinstance(imported, (tuple, list)) else [imported]):
        for item in (group if isinstance(group, (tuple, list)) else [group]):
            if item is not None and hasattr(item, "size") and not hasattr(item, "getAssociatedCloud"):
                clouds.append(item)

    if len(clouds) < 2:
        print(f"  Expected 2 point clouds in {os.path.basename(bin_path)}, found {len(clouds)}")
        return None, None
    if len(clouds) > 2:
        print(f"  {len(clouds)} clouds in {os.path.basename(bin_path)}; matching top/bottom by Pgram number")

    top_pgram = str(top_pgram).strip()
    bot_pgram = str(bot_pgram).strip()

    # Primary: match by embedded Pgram number (only when top != bottom).
    if top_pgram and bot_pgram and top_pgram != bot_pgram:
        top_cloud = next((c for c in clouds if _pgram_of(c) == top_pgram), None)
        bottom_cloud = next((c for c in clouds if _pgram_of(c) == bot_pgram), None)
        if top_cloud is not None and bottom_cloud is not None and top_cloud is not bottom_cloud:
            return top_cloud, bottom_cloud

    # Secondary: 'top'/'bottom' marker on the name (handles top == bottom).
    top_cloud = next((c for c in clouds if _role_from_name(c) == "top"), None)
    bottom_cloud = next((c for c in clouds if _role_from_name(c) == "bottom"), None)
    if top_cloud is not None and bottom_cloud is not None and top_cloud is not bottom_cloud:
        return top_cloud, bottom_cloud

    # Tertiary: fall back to file order.
    print("  WARNING: could not identify top/bottom by name; assuming first cloud "
          "is top, second is bottom.")
    return clouds[0], clouds[1]


def _otsu_threshold(finite_vals):
    """Otsu split of a 1-D distribution: the value maximising between-class
    variance on a 256-bin histogram. Returns the threshold, or None for an
    empty/degenerate distribution so callers can fall back gracefully.
    """
    if len(finite_vals) == 0:
        return None
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    if vmax <= vmin:
        return None

    hist, edges = np.histogram(finite_vals, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)               # cumulative weight of the low class
    w1 = w0[-1] - w0                   # weight of the high class
    cum_mean = np.cumsum(hist * centers)
    mean_total = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = cum_mean / w0
        m1 = (mean_total - cum_mean) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = 0.0
    return float(centers[int(np.argmax(between))])


def trim_mesh_by_distance_to_cloud(mesh, ref_cloud, max_cut_pct=50):
    """Trim the phantom Poisson skirt off an open-surface mesh by each vertex's
    distance to the input point cloud.

    Poisson closes an open surface by extrapolating a "skirt" out past the real
    edge; those phantom vertices sit far from any real sample, while every interior
    vertex — even in a sparsely-sampled patch — stays within ~one sample spacing of
    a point. So thresholding on distance-to-cloud removes only the outer skirt and
    can NEVER punch interior holes (unlike a density threshold).

    The keep/cut split is auto-detected with Otsu on the (bimodal: interior ≈ 0 vs
    skirt ≫ 0) distance distribution, clamped so it never removes more than
    `max_cut_pct` of faces. Falls back to the original mesh on any failure.
    """
    try:
        vert_cloud = mesh.getAssociatedCloud()
        # Distance from each mesh vertex to the nearest real sample point, added as
        # a scalar field on the mesh's vertex cloud in place. Use the EXACT C2C
        # distance (not approx): the approximate version is octree-quantized, so the
        # trim threshold lands on coarse cell boundaries and gives a blocky/staircase
        # edge — exact NN distance varies smoothly, so the trimmed boundary is clean.
        _params = cc.Cloud2CloudDistancesComputationParams()
        _params.maxThreadCount = 0  # auto
        cc.DistanceComputationTools.computeCloud2CloudDistances(vert_cloud, ref_cloud, _params)
        sf_idx = vert_cloud.getNumberOfScalarFields() - 1
        if sf_idx < 0:
            print("  Distance trim: no distance SF produced, skipping")
            return mesh
        vert_cloud.setCurrentOutScalarField(sf_idx)
        sf = vert_cloud.getScalarField(sf_idx)

        d = sf.toNpArrayCopy()
        finite = d[np.isfinite(d)]
        if len(finite) == 0:
            print("  Distance trim: all distances non-finite, skipping")
            return mesh

        otsu_t = _otsu_threshold(finite)
        # Safety cap: keep at least (100 - max_cut_pct)% of vertices — never remove
        # more than max_cut_pct even if Otsu picks too low a split. Clamping the
        # threshold UP only ever leaves more skirt, never eats the interior.
        floor_t = float(np.percentile(finite, 100 - max_cut_pct))
        threshold = max(otsu_t, floor_t) if otsu_t is not None else floor_t
        pct_removed = float((finite > threshold).mean() * 100.0)
        print(f"  Distance trim (auto/Otsu): keep dist<={threshold:.4f} m "
              f"(~{pct_removed:.1f}% of {len(finite)} verts cut, cap {max_cut_pct}%)")

        trimmed = cc.filterBySFValue(0.0, threshold, mesh)
        if trimmed is not None and trimmed.size() > 0:
            print(f"  Distance trim: {mesh.size()} ->{trimmed.size()} faces")
            trimmed.setName(mesh.getName() + "_trimmed")
            return trimmed
        print("  Distance trim: filterBySFValue returned empty/None, using original")
        return mesh
    except Exception as e:
        print(f"  Distance trim failed ({e}) — using original mesh")
        return mesh


def merge_clouds_and_build_mesh(top_cloud, bottom_cloud, su_number):
    print(f"Processing snipped clouds for SU {su_number}...")

    if top_cloud is None or bottom_cloud is None:
        print("Error: missing top or bottom cloud")
        return None, None, None

    top_cloud.setName(f"SU_{su_number}_top")
    bottom_cloud.setName(f"SU_{su_number}_bottom")
    print(f"Top cloud: {top_cloud.size()} points; "
          f"Bottom cloud: {bottom_cloud.size()} points")

    # Per the volume-modeling doc, the merged VOLUME mesh is built from the FULL top
    # + bottom clouds (flip bottom normals -> merge -> Poisson -> solid mesh). We do
    # NOT filter the clouds first: dropping points under-samples the merged cloud so
    # Poisson can't close it and leaves holes in the volume. (A pre-merge c2c-distance
    # filter here is exactly what produced the holey volumes.)

    # Check if clouds have normals
    if not top_cloud.hasNormals():
        print("Warning: Top cloud has no normals, computing...")
        cc.computeNormals([top_cloud])

    if not bottom_cloud.hasNormals():
        print("Warning: Bottom cloud has no normals, computing...")
        cc.computeNormals([bottom_cloud])

    # Invert the normals of the bottom cloud
    print(f"Inverting normals for bottom cloud of SU{su_number}...")
    try:
        cc.invertNormals([bottom_cloud])
    except Exception as e:
        print(f"Error inverting normals: {e}")
        return None, None, None

    # Merge the two clouds
    print("Merging clouds...")
    try:
        merged_cloud = cc.MergeEntities([top_cloud, bottom_cloud],
                                        deleteOriginalClouds=False)
        if merged_cloud is None:
            raise ValueError("Failed to merge clouds")
        print(f"Merged cloud created with {merged_cloud.size()} points")
    except Exception as e:
        print(f"Error merging clouds: {e}")
        return None, None, None

    if not cc.isPluginPoissonRecon():
        print("Error: Poisson Reconstruction plugin not available")
        return merged_cloud, None, None

    # Poisson surface reconstruction with error handling
    merged_mesh = None
    top_mesh = None
    print("Starting Poisson reconstruction for merged cloud...")
    try:
        merged_mesh = cc.PoissonRecon.PR.PoissonReconstruction(
            merged_cloud,
            depth=MERGED_MESH_POISSON_DEPTH,
            density=True,
        )
        merged_mesh.setName(f"SU_{su_number}")
        if merged_mesh is not None:
            # Per the lab's volume-modeling doc, Poisson on the combined cloud should
            # already produce a nice SOLID watertight mesh — so no routine trimming.
            # (The old density-percentile trim here is what punched interior swiss-
            # cheese holes on sparser SUs; large-bubble cases remain a rare manual
            # edit, as the doc notes.)
            print("Mesh created with Poisson Reconstruction.")
        else:
            print("Error: Failed to create mesh")

    except Exception as e:
        print(f"Error during Poisson reconstruction: {e}")

    print("Starting Poisson Reconstruction for top cloud...")

    # Cap the point count fed to the TOP Poisson only (see TOP_MESH_MAX_POINTS). This
    # keeps a badly-cropped/oversized SU from producing a multi-million-face OBJ that
    # hangs the downstream shapefile step. The full top_cloud is used everywhere else
    # (merge, 2.5D volume, distance-trim reference), so volumes are unaffected.
    top_mesh_src = top_cloud
    if top_cloud.size() > TOP_MESH_MAX_POINTS:
        try:
            ref = cc.CloudSamplingTools.subsampleCloudRandomly(top_cloud, TOP_MESH_MAX_POINTS)
            capped = top_cloud.partialClone(ref)[0]  # (ccPointCloud, CLONE_WARNINGS)
            if capped is not None and capped.size() > 0:
                capped.setName(f"SU_{su_number}_top_capped")
                if top_cloud.hasNormals():
                    cc.computeNormals([capped])
                top_mesh_src = capped
                print(f"  Top cloud {top_cloud.size()} pts > cap {TOP_MESH_MAX_POINTS}; "
                      f"using {capped.size()} pts for sheet mesh (volume uses full cloud)")
            else:
                print("  Top-cloud cap produced empty clone; using full top cloud")
        except Exception as e:
            print(f"  Top-cloud cap failed ({e}); using full top cloud")

    try:
        # Reconstruct from the (optionally capped) top cloud — the phantom skirt from
        # the open surface is removed afterwards by distance.
        top_mesh = cc.PoissonRecon.PR.PoissonReconstruction(
            top_mesh_src,
            depth=11,
            density=True,
        )
        top_mesh.setName(f"SU_{su_number}_top_raw")

        if top_mesh is not None:
            print("Mesh created with Poisson Reconstructions.")
        else:
            print("Error: Failed to create top mesh")
    except Exception as e:
        print(f"Error during Poisson reconstruction for top cloud: {e}")

    # Trim the phantom Poisson skirt off the top mesh by distance to the input top
    # cloud (see trim_mesh_by_distance_to_cloud). This replaces the manual step of
    # opening '<su>_post_snip.bin', selecting 'SU_<su>_top_raw' and shrinking its
    # rectangular boundary by hand in CloudCompare. The raw mesh is kept in the
    # project bin so manual fallback is still possible.
    top_mesh_trimmed = top_mesh
    if top_mesh is not None:
        try:
            top_mesh_trimmed = trim_mesh_by_distance_to_cloud(
                top_mesh, top_cloud, max_cut_pct=TOP_MESH_DIST_MAX_CUT_PCT)
            if top_mesh_trimmed is not top_mesh:
                top_mesh_trimmed.setName(f"SU_{su_number}_top")
        except Exception as e:
            print(f"  Top mesh distance trim failed ({e}) — using raw top mesh")
            top_mesh_trimmed = top_mesh

    # Measure 3D volume of the merged mesh
    if merged_mesh is not None:
        try:
            volume_3d, isWarning, stats = cc.ccMesh.computeMeshVolume(merged_mesh)
            # Convert volume_3d from m^3 to cm^3
            volume_3d_cm3 = round(volume_3d * 1e6, 2)

            print(
                f"3D volume of merged mesh: {volume_3d_cm3} cubic centimeters. Warning: {isWarning}"
            )

            report_info = cc.ReportInfoVol()
            success = cc.ComputeVolume25D(
                reportInfo=report_info,
                ground=bottom_cloud,
                ceil=top_cloud,
                vertDim=2,
                gridStep=0.001,
                groundHeight=0.000000,
                ceilHeight=0.000000,
                projectionType=cc.PROJ_AVERAGE_VALUE,
                groundEmptyCellFillStrategy=cc.INTERPOLATE_DELAUNAY,
                groundMaxEdgeLength=0.0,
                ceilEmptyCellFillStrategy=cc.INTERPOLATE_DELAUNAY,
                ceilMaxEdgeLength=0.0,
            )
            volume_25d = report_info.volume if success else ""

            try:
                update_volume_measurements(
                    "volume_measures.txt",
                    su_number,
                    volume_3d_cm3,
                    volume_25d,
                    isWarning,
                )

                print(f"Volume measurements written for {su_number}")
            except Exception as e:
                print(f"Error writing volume measurements: {e}")

        except Exception as e:
            print(f"Error computing volume for merged mesh: {e}")

        # Save the trimmed top mesh as the SU Top OBJ that QGIS reads.
        save_merged_mesh_and_top_mesh(su_number, merged_mesh, top_mesh_trimmed)

        return (
            merged_cloud,
            top_cloud,
            bottom_cloud,
            merged_mesh,
            top_mesh_trimmed,
            top_mesh,  # raw, untrimmed — kept in the project bin for manual fallback
        )


def save_merged_mesh_and_top_mesh(su_number, merged_mesh, top_mesh):
    """Save the merged volume mesh and the top mesh to their destinations.

    The merged volume is written to Data/Final_Volumes/SU_<su>_raw.obj (where the
    dashboard detects post-snip completion) and archived to Volumetrics_<year>/Trench
    NNNNN/SU_<su>.obj. The top mesh is written to Volumetrics_<year>/SU Top OBJs/
    SU_<su>_top.obj, where the Create-SU-Sheet QGIS script reads it.

    Args:
        su_number (str): The SU identifier (e.g. '20001'); also drives the trench folder.
        merged_mesh (cc.Mesh): The merged volume mesh to save.
        top_mesh (cc.Mesh): The top mesh to save.
    """
    # Save the final volume mesh to this season's Volumetrics_<year>/Trench NNNNN/
    # folder as SU_<su>.obj (the lab's archival convention). This is the OBJ the
    # dashboard's Volume button opens. cc.SaveMesh does not create the target
    # directory and fails silently if it's missing, so ensure it exists first.
    trench = _trench_name(su_number)
    if trench:
        trench_dir = os.path.join(_volumetrics_dir(), trench)
        try:
            os.makedirs(trench_dir, exist_ok=True)
            save_path = save_mesh(trench_dir, merged_mesh, file_name=f"SU_{su_number}")
            print(f"Final volume saved for {su_number} at {save_path}")
        except Exception as e:
            print(f"Error saving final volume to Volumetrics: {e}")
    else:
        print(f"  {su_number}: no digits in SU id; cannot resolve trench folder, "
              f"skipping final volume save")

    # Save the top mesh where the Create-SU-Sheet QGIS script (generate_su_sheets.py)
    # reads it: <Volumetrics_<year>>/SU Top OBJs/SU_<su>_top.obj.
    su_top_obj_dir = os.path.join(_volumetrics_dir(), "SU Top OBJs")
    try:
        os.makedirs(su_top_obj_dir, exist_ok=True)
        save_path = save_mesh(
            su_top_obj_dir,
            top_mesh,
            file_name=f"SU_{su_number}_top",
        )
        print(f"Top mesh saved for {su_number} at {save_path}")
    except Exception as e:
        print(f"Error saving top mesh: {e}")


def update_volume_measurements(volume_file, su_number, volume_3d, volume_25d, isWarning):
    notes = "There might be a hole in the mesh" if isWarning else "No issues detected"
    # The first column is written as "SU<su>", so match on the same key — otherwise
    # every re-run appends a duplicate instead of updating the existing row.
    key = f"SU{su_number}"
    new_line = f"{key}\t{volume_3d}\t{volume_25d}\t{notes}.\n"

    lines = []
    if os.path.exists(volume_file):
        with open(volume_file, "r") as vol_file:
            lines = vol_file.readlines()

    # Replace the first existing row for this SU in place, drop any later
    # duplicates, and append if the SU isn't present yet.
    out = []
    found = False
    for line in lines:
        if line.split("\t", 1)[0] == key:
            if not found:
                out.append(new_line)
                found = True
            # else: drop duplicate row for this SU
        else:
            out.append(line)
    if not found:
        out.append(new_line)

    with open(volume_file, "w") as vol_file:
        vol_file.writelines(out)


def run_postsnip_pipeline(json_filepath: str = "input.json") -> None:
    """
    Main entry point for post-snip processing. Input-json-driven: reads su (plus
    top/bottom Pgram numbers) from input.json, finds each SU's combined snip bin
    in Data/SU<su>/, identifies the top and bottom clouds inside it, then merges
    them and builds the volume mesh.

    The operator manually snips each SU in CloudCompare and saves BOTH cropped
    clouds into one project bin named '<su>.bin' (or 'SU<su>.bin', any case).

    Callable from any external Python program:
        import post_snip_script
        post_snip_script.run_postsnip_pipeline("input.json")
    """
    with open(json_filepath) as f:
        job_data = json.load(f)

    produced = 0
    total = len(job_data)
    tarp_progress.report(0, total)
    for i, entry in enumerate(job_data):
        top_pgram = str(entry.get("top", ""))
        bot_pgram = str(entry.get("bottom", ""))
        su = str(entry.get("su", ""))
        if not su or not re.match(r'^[\w\-]+$', su):
            print(f"  Entry {i}: invalid or missing su value {su!r} "
                  f"(must be alphanumeric/hyphen/underscore), skipping")
            tarp_progress.report(i + 1, total, su or '?')
            continue

        bin_path = find_combined_bin(POINT_CLOUD_DIR, su)
        if not bin_path:
            print(
                f"  SU {su}: no combined snip bin found in {POINT_CLOUD_DIR}/SU{su}/. "
                f"Snip the SU in CloudCompare and save both cropped clouds as "
                f"'{su}.bin' (or 'SU{su}.bin') in that folder."
            )
            tarp_progress.report(i + 1, total, su)
            continue

        top_cloud, bottom_cloud = load_top_bottom_from_bin(bin_path, top_pgram, bot_pgram)
        if top_cloud is None or bottom_cloud is None:
            print(
                f"  SU {su}: could not load a top+bottom cloud pair from "
                f"{os.path.basename(bin_path)} (expected two point clouds)."
            )
            tarp_progress.report(i + 1, total, su)
            continue

        print(
            f"\n=== Processing pair {i + 1}/{len(job_data)}: SU {su} ==="
            f"\nBin: {bin_path}\n"
        )

        try:
            result = merge_clouds_and_build_mesh(top_cloud, bottom_cloud, su)
            if result and len(result) >= 4 and result[0] is not None and result[3] is not None:
                merged_cloud, top_cloud, bottom_cloud, merged_mesh, top_mesh, top_mesh_raw = result
                project_path = os.path.join(DATA_DIR, f"SU{su}", f"{su}_post_snip.bin")
                entities = [merged_cloud, top_cloud, bottom_cloud, merged_mesh, top_mesh]
                # Keep the raw, untrimmed top mesh in the bin too (manual fallback),
                # but only when the trim actually produced a distinct mesh.
                if top_mesh_raw is not None and top_mesh_raw is not top_mesh:
                    entities.append(top_mesh_raw)
                save_project(entities, project_path)
                print(f"Successfully finished processing SU {su} and saved project at {project_path}")
                produced += 1
            else:
                print(f"  SU {su}: processing failed (merge or mesh returned None)")
            tarp_progress.report(i + 1, total, su)

        except Exception as e:
            print(f"  Unexpected error processing SU {su}: {e}")
            tarp_progress.report(i + 1, total, su)
            continue

    if produced == 0 and job_data:
        raise RuntimeError(
            "No combined snip bins found for any SU in this run. For each SU, snip "
            "the top and bottom clouds in CloudCompare and save BOTH into a single "
            "project bin named '<su>.bin' (or 'SU<su>.bin') inside Data/SU<su>/."
        )


if __name__ == "__main__":
    run_postsnip_pipeline(sys.argv[1] if len(sys.argv) > 1 else "input.json")
