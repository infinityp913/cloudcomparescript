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

# The top mesh's phantom skirt is trimmed by auto-detecting the split between the
# low-density skirt and the high-density real surface in the Poisson density SF
# (Otsu's method — see _auto_density_threshold). This replaces the manual step of
# tightening the density SF display range by hand on '<su>_top_raw' in CloudCompare.
#
# Safety cap: the auto threshold is clamped so it never removes more than this
# percentage of faces. Otsu finds the skirt/surface valley on its own, but on an
# (almost) skirt-free mesh it would still split the real surface — the cap keeps it
# from eating good geometry in that case. Raise if a real skirt survives; lower if
# the surface gets clipped. Prefer leaving a little skirt over losing real surface.
TOP_MESH_DENSITY_MAX_CUT_PCT = 15



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


def _auto_density_threshold(finite_vals, max_cut_pct=15):
    """Auto-detect the density value separating the low-density skirt from the
    high-density real surface, via Otsu's method on the density histogram.

    Returns the threshold (keep values >= it), clamped so it never removes more
    than `max_cut_pct` percent of the values. Returns None if a threshold can't
    be computed (empty/degenerate distribution), so callers fall back gracefully.
    """
    if len(finite_vals) == 0:
        return None
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    if vmax <= vmin:
        return None

    # Otsu: pick the histogram split maximising between-class variance.
    hist, edges = np.histogram(finite_vals, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)                       # weight of the low (skirt) class
    w1 = w0[-1] - w0                           # weight of the high (surface) class
    cum_mean = np.cumsum(hist * centers)
    mean_total = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = cum_mean / w0
        m1 = (mean_total - cum_mean) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = 0.0
    otsu_t = float(centers[int(np.argmax(between))])

    # Clamp: never cut more than max_cut_pct% of faces (protects the real surface
    # when the distribution is ~unimodal and Otsu would split it arbitrarily).
    cap_t = float(np.percentile(finite_vals, max_cut_pct))
    return min(otsu_t, cap_t)


def trim_mesh_by_density(mesh, density_min_pct=10, auto=False, max_cut_pct=15):
    """Remove low-density Poisson faces (phantom boundary bubbles / skirt).

    Poisson reconstruction with density=True assigns a per-vertex density
    scalar. Phantom faces at the crop boundary have few real points nearby
    and thus low density.  Filtering by density removes these artifacts.

    With auto=False the cut is a fixed `density_min_pct` percentile (used for the
    merged volume mesh). With auto=True the skirt/surface split is auto-detected
    via Otsu and clamped to at most `max_cut_pct` of faces (used for the open top
    mesh, whose skirt fraction varies per SU). Falls back to the original mesh if
    the API call fails.
    """
    try:
        vert_cloud = mesh.getAssociatedCloud()
        n_sf = vert_cloud.getNumberOfScalarFields()
        if n_sf == 0:
            print("  Density trim: no SFs on vertex cloud, skipping")
            return mesh

        density_sf_idx = 0
        vert_cloud.setCurrentScalarField(density_sf_idx)
        sf = vert_cloud.getScalarField(density_sf_idx)
        print(f"  Density trim: SF='{sf.getName()}' "
              f"range [{sf.getMin():.2f}, {sf.getMax():.2f}]")

        vals = sf.toNpArrayCopy()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return mesh

        if auto:
            threshold = _auto_density_threshold(finite, max_cut_pct=max_cut_pct)
            if threshold is None:
                print("  Density trim (auto): degenerate distribution, skipping")
                return mesh
            pct_removed = float((finite < threshold).mean() * 100.0)
            print(f"  Density trim (auto/Otsu): threshold={threshold:.3f} "
                  f"(~{pct_removed:.1f}% of {len(finite)} verts, cap {max_cut_pct}%)")
        else:
            threshold = float(np.percentile(finite, density_min_pct))
            print(f"  Density trim: threshold={threshold:.3f} "
                  f"(p{density_min_pct} of {len(finite)} vertices)")

        sf_max = float(sf.getMax())
        trimmed = cc.filterBySFValue(threshold, sf_max * 1.01, mesh)

        if trimmed is not None and trimmed.size() > 0:
            print(f"  Density trim: {mesh.size()} ->{trimmed.size()} faces")
            trimmed.setName(mesh.getName() + "_trimmed")
            return trimmed
        else:
            print("  Density trim: filterBySFValue returned empty/None, using original")
            return mesh
    except Exception as e:
        print(f"  Density trim failed ({e}) — using original mesh")
        return mesh


def filter_by_c2c_distance(cloud, low_percentile=10):
    """
    Remove points whose C2C distance scalar field value falls below
    `low_percentile` percent of the distribution. These low-distance fringe
    points sit where the top/bottom surfaces nearly meet at the SU boundary
    and cause Poisson to fill in phantom bubble surfaces.

    Returns a filtered clone, or the original cloud if filtering isn't possible.
    """
    n_sf = cloud.getNumberOfScalarFields()
    if n_sf == 0:
        print(f"  C2C filter: no scalar fields on {cloud.getName()}, skipping")
        return cloud

    # The C2C distance field is the first scalar field added by pre_snip
    sf_idx = 0
    cloud.setCurrentOutScalarField(sf_idx)
    sf = cloud.getScalarField(sf_idx)
    sf_name = sf.getName()

    vals = sf.toNpArrayCopy()
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        print(f"  C2C filter: all values non-finite on {cloud.getName()}, skipping")
        return cloud

    threshold = float(np.percentile(finite, low_percentile))
    sf_max    = float(finite.max())
    print(f"  C2C filter '{sf_name}': keeping >{threshold:.4f} m "
          f"(p{low_percentile} of [{finite.min():.4f}, {sf_max:.4f}])")

    filtered = cc.filterBySFValue(threshold, sf_max * 1.01, cloud)
    if filtered is None or filtered.size() == 0:
        print(f"  C2C filter: result empty, using original {cloud.getName()}")
        return cloud

    filtered.setName(cloud.getName() + "_c2c_filtered")
    print(f"  C2C filter: {cloud.size()} ->{filtered.size()} pts")
    return filtered


def merge_clouds_and_build_mesh(top_cloud, bottom_cloud, su_number):
    print(f"Processing snipped clouds for SU {su_number}...")

    if top_cloud is None or bottom_cloud is None:
        print("Error: missing top or bottom cloud")
        return None, None, None

    top_cloud.setName(f"SU{su_number}_top")
    bottom_cloud.setName(f"SU{su_number}_bottom")
    print(f"Top cloud: {top_cloud.size()} points; "
          f"Bottom cloud: {bottom_cloud.size()} points")

    # Filter out low-C2C-distance fringe points that cause Poisson bubble artifacts
    top_cloud    = filter_by_c2c_distance(top_cloud,    low_percentile=25)
    bottom_cloud = filter_by_c2c_distance(bottom_cloud, low_percentile=25)

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
        # Try with lower depth first to avoid memory issues
        merged_mesh = cc.PoissonRecon.PR.PoissonReconstruction(
            merged_cloud,
            depth=11,
            density=True,
        )
        merged_mesh.setName(f"{su_number}_merged")
        if merged_mesh is not None:
            print("Mesh created with Poisson Reconstruction.")
            merged_mesh = trim_mesh_by_density(merged_mesh, density_min_pct=10)
        else:
            print("Error: Failed to create mesh")

    except Exception as e:
        print(f"Error during Poisson reconstruction: {e}")

    print("Starting Poisson Reconstruction for top cloud...")

    try:
        top_mesh = cc.PoissonRecon.PR.PoissonReconstruction(
            top_cloud,
            depth=11,
            density=True,
        )
        top_mesh.setName(f"{su_number}_top_raw")

        if merged_mesh is not None:
            print("Mesh created with Poisson Reconstructions.")
        else:
            print("Error: Failed to create mesh")
    except Exception as e:
        print(f"Error during Poisson reconstruction for top cloud: {e}")

    # Trim the phantom Poisson skirt off the top mesh the same way the merged
    # mesh is trimmed above. This replaces the manual CloudCompare step where the
    # operator opens '<su>_post_snip.bin', selects '<su>_top_raw' and shrinks its
    # rectangular boundary by adjusting the density SF display range. The raw mesh
    # is kept in the project bin so that manual fallback is still possible.
    top_mesh_trimmed = top_mesh
    if top_mesh is not None:
        try:
            top_mesh_trimmed = trim_mesh_by_density(
                top_mesh, auto=True, max_cut_pct=TOP_MESH_DENSITY_MAX_CUT_PCT)
            if top_mesh_trimmed is not top_mesh:
                top_mesh_trimmed.setName(f"{su_number}_top_trimmed")
        except Exception as e:
            print(f"  Top mesh density trim failed ({e}) — using raw top mesh")
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
