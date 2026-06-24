import json
import cloudComPy as cc
import cloudComPy.PoissonRecon
import gc
import os
import sys
import math

from pre_snip_script import load_cloud, save_mesh, DATA_DIR, save_project
import glob

json_filepath = sys.argv[1] if len(sys.argv) > 1 else "example.json"
POINT_CLOUD_DIR = "Data"

with open(json_filepath, "r") as f:
    job_data = json.load(f)


def get_job_number_from_filename(filename):
    """
    Extract the job number from the filename.
    The job number is expected to be in the format "Pgram_Job_<job_number>_SU...".

    Args:
        filename (str): The filename to extract the job number from.

    Returns:
        str: The extracted job number, or None if not found.
    """
    parts = filename.lower().split("pgram_job_")
    if len(parts) > 1:
        job_part = parts[1].split("_")[0]
        if job_part.isdigit():
            return job_part
    print(f"Warning: Job number not found in filename {filename}")
    return None


def get_su_number_from_filename(filename):
    """
    Extract the SU number from the filename.
    The SU number is expected to be in the format "cleaned_su_<su_number>.bin".

    Args:
        filename (str): The filename to extract the SU number from.

    Returns:
        str: The extracted SU number, or None if not found.
    """
    parts = filename.lower().split("_cleaned_su_")
    if len(parts) > 1:
        su_part = parts[1].replace(".bin", "")
        if su_part:
            return su_part
    print(f"Warning: SU number not found in filename {filename}")
    return None


def find_top_bottom_cloud_pairs(point_cloud_dir):
    """
    Find pairs of top and bottom point clouds in the given directory.
    Each pair consists of two point clouds with the same prefix (Pgram_Job_<job_number>_<comma separated list of SUs>_cleaned_su_) and a unique suffix (the su number).

    The function searches for subdirectories that match the top names from the job data JSON file.
    It then looks for files in those subdirectories that match the pattern *_cleaned_su_*.bin.
    For each unique su number found, it pairs the top and bottom clouds based on their prefixes.

    A HUGE assumption that the function makes: the top cloud will always have a lower job number than the bottom cloud.

    The function ensures that the top cloud has a lower job number than the bottom cloud to avoid duplicates.
    The pairs are returned as a list of tuples, where each tuple contains the full paths to the top and bottom clouds.

    Args:
        point_cloud_dir (str): The directory containing the point clouds.
    Returns:
        list: A list of tuples, where each tuple contains the full paths to the top and
    """

    pairs = []
    for subdir in os.listdir(point_cloud_dir):
        # Only consider subdirs that contain job["top"] for job in job_data
        top_names_from_json = set(job["top"] for job in job_data if "top" in job)
        for top_name_from_json in top_names_from_json:
            if top_name_from_json.lower() in subdir.lower():
                print(
                    f"Found matching subdir for top name from json file {top_name_from_json}: {subdir} --- searching for cleaned bin files"
                )
                subdir_path = os.path.join(point_cloud_dir, subdir)
                # Find all *_cleaned_su_*.bin files in the subdir
                bin_files = glob.glob(os.path.join(subdir_path, "*_cleaned_su_*.bin"))
                # Map from (prefix, su) to full path
                clouds = {}
                for f in bin_files:
                    base = os.path.basename(f)
                    # Extract prefix and su number
                    if "_cleaned_su_" in base.lower() and base.lower().endswith(".bin"):
                        prefix = base.split("_cleaned_su_")[0]
                        su = base.split("_cleaned_su_")[1].replace(".bin", "")
                        if su.endswith("_top"):
                            su = su.replace("_top", "")
                            prefix = prefix + "_top"
                        clouds[(prefix, su)] = f
                # For each su, if there are at least two clouds, consider all pairs
                su_numbers = set(su for (_, su) in clouds.keys())
                for su in su_numbers:
                    prefixes = [prefix for (prefix, s) in clouds.keys() if s == su]
                    if len(prefixes) >= 2:
                        for i in range(len(prefixes)):
                            for j in range(len(prefixes)):
                                if i != j and prefixes[i].endswith("_top") and not prefixes[j].endswith("_top"):
                                    top = prefixes[i]
                                    bottom = prefixes[j]
                                    pairs.append(
                                        (clouds[(top, su)], clouds[(bottom, su)])
                                    )
                                elif i != j and not prefixes[i].endswith("_top") and prefixes[j].endswith("_top"):
                                    top = prefixes[i]
                                    bottom = prefixes[j]
                                    pairs.append(
                                        (clouds[(top, su)], clouds[(bottom, su)])
                                    )
                                else:
                                    job_number_i = get_job_number_from_filename(prefixes[i])
                                    job_number_j = get_job_number_from_filename(prefixes[j])
                                    if i != j and job_number_i < job_number_j:
                                        top = prefixes[i]
                                        bottom = prefixes[j]
                                        pairs.append(
                                            (clouds[(top, su)], clouds[(bottom, su)])
                                        )
                    print(
                    f"Found {len(pairs)} pairs of top and bottom clouds for {subdir}."
                )

    return pairs


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


def merge_clouds_and_build_mesh(top_cloud_path, bottom_cloud_path):
    top_base_name = os.path.basename(top_cloud_path).split("_cleaned_su_")[0]
    bottom_base_name = os.path.basename(bottom_cloud_path).split("_cleaned_su_")[0]

    su_number = get_su_number_from_filename(top_cloud_path)

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
        merged_cloud = cc.MergeEntities(
            [top_cloud, bottom_cloud], deleteOriginalClouds=False
        )
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
            print("Mesh created with Poisson Reconstructions.")
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


def update_volume_measurements(volume_file, su_name, volume_3d, volume_25d, isWarning):
    notes = "There might be a hole in the mesh" if isWarning else "No issues detected"
    new_line = f"SU{su_name}\t{volume_3d}\t{volume_25d}\t{notes}.\n"

    # Read all lines and check if su_name exists
    lines = []
    found = False
    if os.path.exists(volume_file):
        with open(volume_file, "r") as vol_file:
            lines = vol_file.readlines()
        for idx, line in enumerate(lines):
            if line.split("\t", 1)[0] == su_name:
                lines[idx] = new_line
                found = True
                break

    if not found:
        lines.append(new_line)

    with open(volume_file, "w") as vol_file:
        vol_file.writelines(lines)


if __name__ == "__main__":
    # Find all top and bottom cloud pairs
    pairs = find_top_bottom_cloud_pairs(POINT_CLOUD_DIR)

    for i, (top_cloud_path, bottom_cloud_path) in enumerate(pairs):
        top_base_name = os.path.basename(top_cloud_path).split("_cleaned_su_")[0]
        su_number = top_cloud_path.split("_cleaned_su_")[1].replace(".bin", "")

        print(
            f"\n=== Processing pair {i + 1}/{len(pairs)}: {su_number} ==="
            f"\nTop Cloud: {top_cloud_path}\nBottom Cloud: {bottom_cloud_path}\n"
        )

        try:
            # Call the function to merge clouds and build mesh
            (
                merged_cloud,
                top_cloud,
                bottom_cloud,
                merged_mesh,
                top_mesh,
                # top_mesh_filtered,
            ) = merge_clouds_and_build_mesh(top_cloud_path, bottom_cloud_path)

            if merged_cloud is not None and merged_mesh is not None:
                project_name = f"{top_base_name}/{su_number}_post_snip.bin"
                project_path = os.path.join(DATA_DIR, project_name)
                save_project(
                    [
                        merged_cloud,
                        top_cloud,
                        bottom_cloud,
                        merged_mesh,
                        top_mesh,
                        # top_mesh_filtered,
                    ],
                    project_path,
                )
                print(
                    f"Successfully finished processing {su_number} and saved project at {project_path}"
                )
            else:
                print(f"Failed to process {su_number}")

        except Exception as e:
            print(f"Unexpected error processing {su_number}: {e}")
            continue
