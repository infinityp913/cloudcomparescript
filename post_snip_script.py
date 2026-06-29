import json
import re
import cloudComPy as cc
import cloudComPy.PoissonRecon
import gc
import os
import sys
import math
import numpy as np

from pre_snip_script import load_cloud, save_mesh, DATA_DIR, save_project, find_mesh_by_pgram_job
import glob
import tarp_progress

POINT_CLOUD_DIR = "Data"



def find_top_bottom_cloud_pairs(point_cloud_dir, top_id):
    """
    Find the manually-snipped bin pair for one SU in Data/<top_id>/.
    Returns (top_path, bottom_path) or (None, None) if not found.
    Picks newest by mtime when multiple candidates exist (re-crop case).
    """
    folder = os.path.join(point_cloud_dir, top_id)
    if not os.path.isdir(folder):
        print(f"  Folder not found: {folder}")
        return None, None

    top_candidates = glob.glob(os.path.join(folder, "*_top_with_dist_*_snipped.bin"))
    bot_candidates = glob.glob(os.path.join(folder, "*_bottom_with_dist_*_snipped.bin"))

    if not top_candidates or not bot_candidates:
        return None, None

    top_path = max(top_candidates, key=os.path.getmtime)
    bot_path = max(bot_candidates, key=os.path.getmtime)

    if len(top_candidates) > 1:
        print(f"  Multiple _snipped top bins; using newest: {os.path.basename(top_path)}")
    if len(bot_candidates) > 1:
        print(f"  Multiple _snipped bottom bins; using newest: {os.path.basename(bot_path)}")

    return top_path, bot_path


def trim_mesh_by_density(mesh, density_min_pct=10):
    """Remove low-density Poisson faces (phantom boundary bubbles).

    Poisson reconstruction with density=True assigns a per-vertex density
    scalar. Phantom faces at the crop boundary have few real points nearby
    and thus low density.  Filtering by density removes these artifacts.
    Falls back to the original mesh if the API call fails.
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

        vals = sf.toNpArray()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return mesh
        threshold = float(np.percentile(finite, density_min_pct))
        sf_max = float(sf.getMax())
        print(f"  Density trim: threshold={threshold:.3f} "
              f"(p{density_min_pct} of {len(finite)} vertices)")

        trimmed = cc.filterBySFValue(threshold, sf_max * 1.01, mesh)

        if trimmed is not None and trimmed.size() > 0:
            print(f"  Density trim: {mesh.size()} → {trimmed.size()} faces")
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

    import numpy as np
    vals = sf.toNpArray()
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
    print(f"  C2C filter: {cloud.size()} → {filtered.size()} pts")
    return filtered


def merge_clouds_and_build_mesh(top_cloud_path, bottom_cloud_path, su_number, top_base_name, bottom_base_name):
    print(f"Loading point clouds for {top_base_name} and {bottom_base_name}...")

    # Load clouds with error checking
    try:
        print(f"Loading file {bottom_cloud_path}")
        top_cloud = load_cloud(f"{top_cloud_path}", label=f"{top_base_name}_top")
        if top_cloud is None:
            raise ValueError(f"Failed to load top cloud: {top_base_name}")
        print(f"Top cloud loaded: {top_cloud.size()} points")

        bottom_cloud = load_cloud(
            f"{bottom_cloud_path}", label=f"{bottom_base_name}_bottom"
        )
        if bottom_cloud is None:
            raise ValueError(f"Failed to load bottom cloud: {bottom_base_name}")
        print(f"Bottom cloud loaded: {bottom_cloud.size()} points")

    except Exception as e:
        print(f"Error loading clouds: {e}")
        return None, None, None

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
    print(f"Inverting normals for bottom cloud of {bottom_base_name}...")
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

    # # filter the top cloud by density and run poisson reconstruction on it
    # if top_cloud is not None:
    #     if top_cloud.getNumberOfScalarFields() > 0:
    #         try:
    #             # filter out the points of top_cloud by density filtering out anything below 1 standard deviation away from the mean
    #             sfc = top_cloud.getScalarField(top_cloud.getNumberOfScalarFields() - 1)
    #             sf_mean, sf_variance = sfc.computeMeanAndVariance()
    #             top_cloud_filtered = cc.filterBySFValue(
    #                 float(sf_mean - math.sqrt(sf_variance)),
    #                 float(sfc.getMax()),
    #                 top_cloud.cloneThis(),
    #             )
    #             top_cloud_filtered.setName(f"{su_number}_top_cloud_filtered")
    #             print(
    #                 f"Filtered top cloud created with {top_cloud_filtered.size()} points."
    #             )
    #         except Exception as e:
    #             print(f"Error filtering top cloud: {e}")
    #             top_cloud_filtered = None

    #         try:
    #             if top_cloud_filtered is not None:
    #                 top_mesh_filtered = cc.PoissonRecon.PR.PoissonReconstruction(
    #                     top_cloud_filtered,
    #                     depth=11,
    #                     density=True,
    #                 )
    #                 top_mesh_filtered.setName(f"{su_number}_top_filtered")
    #                 if top_mesh_filtered is None:
    #                     print(
    #                         "Warning: Poisson reconstruction for filtered top cloud returned None, trying with different parameters..."
    #                     )
    #                     # Try again with even lower depth
    #                     top_mesh_filtered = cc.PoissonRecon.PR.PoissonReconstruction(
    #                         top_cloud_filtered, depth=8, density=True
    #                     )
    #                 else:
    #                     top_mesh_filtered.setName(f"{su_number}_top_filtered")
    #                     print(
    #                         "Mesh created for filtered top cloud with Poisson Reconstructions."
    #                     )
    #             else:
    #                 print("No filtered top cloud available, skipping Poisson reconstruction.")
    #                 top_mesh_filtered = None
    #         except Exception as e:
    #             print(f"Error during Poisson reconstruction for filtered top cloud: {e}")
    #             top_mesh_filtered = None
    #     else:
    #         print(
    #             "Warning: Top cloud has no scalar fields, skipping filtering and Poisson reconstruction."
    #         )
    #         top_mesh_filtered = None
    # else:
    #     print("No top cloud available, skipping filtering and Poisson reconstruction.")
    #     top_mesh_filtered = None

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

        save_merged_mesh_and_top_mesh(
            su_number, top_base_name, merged_mesh, top_mesh
        )

        return (
            merged_cloud,
            top_cloud,
            bottom_cloud,
            merged_mesh,
            top_mesh,
            # top_mesh_filtered,
        )


def save_merged_mesh_and_top_mesh(
    su_number, top_base_name, merged_mesh, top_mesh
):
    """Save the merged mesh and top mesh to the specified directory.
    Args:
        su_number (str): The identifier for the surface.
        top_base_name (str): The base name for the top mesh. This is used to save the meshes in the right folder named after the top cloud's full Pgram+SU name.
        merged_mesh (cc.Mesh): The merged mesh to save.
        top_mesh (cc.Mesh): The top mesh to save.
    """
    # Save merged mesh
    try:
        save_mesh(
            f"{POINT_CLOUD_DIR}/Final_Volumes", merged_mesh, file_name=f"SU_{su_number}_raw"
        )
        print(
            f"Mesh saved for {su_number} at {POINT_CLOUD_DIR}/Final_Volumes/SU_{su_number}_raw.obj"
        )
    except Exception as e:
        print(f"Error saving mesh: {e}")

    # Save top mesh
    try:
        save_mesh(
            f"{POINT_CLOUD_DIR}/{top_base_name}",
            top_mesh,
            file_name=f"SU_{su_number}_top_raw",
        )
        print(
            f"Top mesh saved for {su_number} at {POINT_CLOUD_DIR}/{top_base_name}/SU_{su_number}_top_raw.obj"
        )
    except Exception as e:
        print(f"Error saving top mesh: {e}")

    # # Save filtered top mesh if it exists
    # if top_mesh_filtered:
    #     try:
    #         save_mesh(
    #             f"{POINT_CLOUD_DIR}/Final_Volume_Tops",
    #             top_mesh_filtered,
    #             file_name=f"SU_{su_number}_top_filtered",
    #         )
    #         print(
    #             f"Filtered top mesh saved for {su_number} at {POINT_CLOUD_DIR}/Final_Volume_Tops/SU_{su_number}_top_filtered.obj"
    #         )
    #     except Exception as e:
    #         print(f"Error saving filtered top mesh: {e}")

    # Force garbage collection to free memory
    # gc.collect()


def update_volume_measurements(volume_file, su_number, volume_3d, volume_25d, isWarning):
    notes = "There might be a hole in the mesh" if isWarning else "No issues detected"
    new_line = f"SU{su_number}\t{volume_3d}\t{volume_25d}\t{notes}.\n"

    # Read all lines and check if su_number exists
    lines = []
    found = False
    if os.path.exists(volume_file):
        with open(volume_file, "r") as vol_file:
            lines = vol_file.readlines()
        for idx, line in enumerate(lines):
            if line.split("\t", 1)[0] == su_number:
                lines[idx] = new_line
                found = True
                break

    if not found:
        lines.append(new_line)

    with open(volume_file, "w") as vol_file:
        vol_file.writelines(lines)


def run_postsnip_pipeline(json_filepath: str = "input.json") -> None:
    """
    Main entry point for post-snip processing. Input-json-driven: reads top/bottom/su
    from input.json, resolves the PLY stems via find_mesh_by_pgram_job, then globs
    Data/<top_id>/ for manually-snipped *_snipped.bin pairs saved by the operator
    after cropping in CloudCompare.

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
        top_pgram = entry.get("top", "")
        bot_pgram = entry.get("bottom", "")
        su = str(entry.get("su", ""))
        if su and not re.match(r'^[\w\-]+$', su):
            print(f"  Entry {i}: invalid su value {su!r} (must be alphanumeric/hyphen/underscore), skipping")
            tarp_progress.report(i + 1, total, su)
            continue

        top_id = find_mesh_by_pgram_job(top_pgram)
        bot_id = find_mesh_by_pgram_job(bot_pgram)

        if not top_id or not bot_id:
            print(f"  SU {su or '?'}: could not resolve PLY stem for top={top_pgram!r} or bottom={bot_pgram!r}, skipping")
            tarp_progress.report(i + 1, total, su or '?')
            continue

        # Fallback: parse su from the top_id stem when the JSON entry lacks it
        if not su:
            parts = top_id.split("_SU_")
            su = parts[1].split("_")[0] if len(parts) > 1 else ""

        top_path, bot_path = find_top_bottom_cloud_pairs(POINT_CLOUD_DIR, top_id)
        if not top_path or not bot_path:
            print(
                f"  SU {su}: no manually-snipped bins found in {POINT_CLOUD_DIR}/{top_id}/. "
                f"Open the pre-snip bins in CC, crop top & bottom, and Save As with a "
                f"`_snipped` suffix in the same folder."
            )
            tarp_progress.report(i + 1, total, su)
            continue

        print(
            f"\n=== Processing pair {i + 1}/{len(job_data)}: SU {su} ==="
            f"\nTop:    {top_path}\nBottom: {bot_path}\n"
        )

        try:
            result = merge_clouds_and_build_mesh(top_path, bot_path, su, top_id, bot_id)
            if result and result[0] is not None and result[3] is not None:
                merged_cloud, top_cloud, bottom_cloud, merged_mesh, top_mesh = result
                project_path = os.path.join(DATA_DIR, top_id, f"{su}_post_snip.bin")
                save_project(
                    [merged_cloud, top_cloud, bottom_cloud, merged_mesh, top_mesh],
                    project_path,
                )
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
            "No manually-snipped clouds found for any SU in this run. "
            "Open each SU's pre-snip bins (Open in CC), crop top & bottom, and Save As "
            "with a `_snipped` suffix in the same `Data/<Pgram_Job_...>` folder before "
            "running post-snip."
        )


if __name__ == "__main__":
    run_postsnip_pipeline(sys.argv[1] if len(sys.argv) > 1 else "input.json")
