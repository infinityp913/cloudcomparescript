import cloudComPy as cc
import json
import numpy as np
import scipy
import requests
import matplotlib
import os
import sys

INPUT_MESH_PATH = os.path.expanduser("~/Documents/TARP/ply/")
DATA_DIR = os.path.expanduser("./Data")

os.makedirs(DATA_DIR, exist_ok=True)


def load_mesh(mesh_dir: str, job_id: str) -> cc.MESH:
    """
    Load a mesh from a file based on the job ID.
    Args:
        job_id (str): The ID of the job to load the mesh for.
    Returns:
        cc.Mesh: The loaded mesh object.
    Raises:
        ValueError: If the job ID is empty.
        TypeError: If the job ID is not a string.
    """
    if not job_id:
        raise ValueError("Job ID cannot be empty")
    if not isinstance(job_id, str):
        raise TypeError("Job ID must be a string")
    ply_file_path = os.path.join(mesh_dir, f"{job_id}.ply")
    mesh = cc.loadMesh(ply_file_path)
    mesh.setName(job_id)
    print(f"Loaded  and set name for mesh: {mesh.getName()}")
    return mesh


def load_cloud(cloud_path: str, label: str) -> cc.POINT_CLOUD:
    """
    Load a point cloud from a file based on the job ID.
    Args:
        job_id (str): The ID of the job to load the point cloud for.
    Returns:
        cc.PointCloud: The loaded point cloud object.
    Raises:
        ValueError: If the job ID is empty.
        TypeError: If the job ID is not a string.
    """
    mesh = cc.loadPointCloud(cloud_path)
    mesh.setName(label)
    return mesh


def sample_mesh(mesh: cc.MESH, density=20000) -> cc.POINT_CLOUD:
    """
    Sample a mesh to create a point cloud.
    Args:
        mesh (cc.MESH): The mesh to sample.
        density (int): The number of points to sample from the mesh.
    Raises:
        ValueError: If the mesh is empty or not a valid mesh.
        TypeError: If the mesh is not an instance of cc.MESH
    """
    cloud = mesh.samplePoints(
        densityBased=True, samplingParameter=density, withNormals=True
    )
    cloud.setName(mesh.getName())
    return cloud


def save_cloud(output_dir, cloud, name_suffix=None, file_name=None):
    if name_suffix is not None:
        name = f"{cloud.getName()}_{name_suffix}" if name_suffix else cloud.getName()
    elif file_name is not None:
        name = file_name
    save_path = os.path.join(output_dir, f"{name}.bin")
    res = cc.SavePointCloud(cloud, save_path)
    return save_path


def save_mesh(output_dir, mesh, name_suffix=None, file_name=None):
    if name_suffix is not None:
        name = f"{mesh.getName()}_{name_suffix}" if name_suffix else mesh.getName()
    elif file_name is not None:
        name = file_name
    save_path = os.path.join(output_dir, f"{name}.obj")
    res = cc.SaveMesh(mesh, save_path)
    return save_path


def compute_bidirectional_distances(cloud1, cloud2) -> tuple:
    """Compute bidirectional distances between two point clouds.
    Args:
        cloud1: The first point cloud.
        cloud2: The second point cloud.
    Returns:
        tuple: A tuple containing two point clouds with computed distances.
    Raises:
        ValueError: If either cloud is empty or does not have normals.
    """
    if cloud1.size() == 0 or cloud2.size() == 0:
        raise ValueError("Both point clouds must be non-empty")

    cloud1_copy = cloud1.cloneThis()
    cloud2_copy = cloud2.cloneThis()
    cc.DistanceComputationTools.computeApproxCloud2CloudDistance(cloud1_copy, cloud2)
    cc.DistanceComputationTools.computeApproxCloud2CloudDistance(cloud2_copy, cloud1)
    return cloud1_copy, cloud2_copy


def compute_detailed_cloud_distances(
    compared_cloud,
    reference_cloud,
    octree_level=7,
    max_distance=0.524650,
    local_model="QUADRIC",
    knn_points=6,
    sphere_radius=0.083536,
    use_spherical_search=False,
    split_xyz=False,
    multi_threaded=True,
    max_thread_count=7,
    reset_former_distances=True,
    reuse_existing_local_models=False,
) -> tuple:
    """
    Compute detailed cloud-to-cloud distances using CloudCompare's computeCloud2CloudDistances.

    Args:
        compared_cloud: The compared point cloud (mesh.sampled in your case)
        reference_cloud: The reference point cloud (mesh.sampled in your case)
        octree_level (int): Octree subdivision level (default: 7)
        max_distance (float): Maximum distance threshold (default: 0.524650)
        local_model (str): Local modeling method - 'QUADRIC', 'LS', 'TRI', 'NO_MODEL' (default: 'QUADRIC')
        knn_points (int): Number of nearest neighbors for local modeling (default: 6)
        sphere_radius (float): Sphere radius for local modeling (default: 0.083536)
        use_spherical_search (bool): Use spherical search instead of kNN (default: False)
        split_xyz (bool): Split X,Y and Z components (default: False)
        multi_threaded (bool): Enable multi-threading (default: True)
        max_thread_count (int): Maximum thread count (default: 7)
        reset_former_distances (bool): Reset existing distances (default: True)
        reuse_existing_local_models (bool): Reuse existing local models for speed (default: False)

    Returns:
        tuple: A tuple containing (compared_cloud_with_distances, reference_cloud_copy, result_code)

    Raises:
        ValueError: If either cloud is empty or parameters are invalid.
    """

    # Validate input clouds
    if compared_cloud.size() == 0 or reference_cloud.size() == 0:
        raise ValueError("Both point clouds must be non-empty")

    # Validate parameters
    if octree_level < 0 or octree_level > 21:
        raise ValueError("Octree level must be between 0 and 21 (0 = auto)")

    if max_distance < 0:
        raise ValueError("Max distance must be non-negative (-1 to deactivate)")

    valid_models = ["QUADRIC", "LS", "TRI", "NO_MODEL"]
    if local_model not in valid_models:
        raise ValueError(f"Local model must be one of {valid_models}")

    if knn_points < 3:
        raise ValueError("KNN points must be at least 3")

    if sphere_radius <= 0:
        raise ValueError("Sphere radius must be positive")

    # Create copies to avoid modifying original clouds
    compared_copy = compared_cloud.cloneThis()
    reference_copy = reference_cloud.cloneThis()

    # Create parameters object
    params = cc.Cloud2CloudDistancesComputationParams()

    # Set general parameters
    params.octreeLevel = octree_level
    params.maxSearchDist = max_distance if max_distance > 0 else -1
    params.multiThread = multi_threaded
    params.maxThreadCount = max_thread_count if multi_threaded else 0
    params.resetFormerDistances = reset_former_distances

    # Set local modeling parameters using the correct enum values
    # Access the enum values through CCCoreLib or cc module
    try:
        # Try to access enum values - adjust the module path as needed for your CloudComPy installation
        if local_model == "NO_MODEL":
            params.localModel = cc.LOCAL_MODEL_TYPES.NO_MODEL
        elif local_model == "LS":
            params.localModel = cc.LOCAL_MODEL_TYPES.LS
        elif local_model == "TRI":
            params.localModel = cc.LOCAL_MODEL_TYPES.TRI
        elif local_model == "QUADRIC":
            params.localModel = cc.LOCAL_MODEL_TYPES.QUADRIC
        print("Set localModel using cc.LOCAL_MODEL_TYPES")
    except AttributeError:
        # If the above doesn't work, try alternative access patterns
        try:
            # Alternative 1: Through CCCoreLib
            import CCCoreLib

            if local_model == "NO_MODEL":
                params.localModel = CCCoreLib.LOCAL_MODEL_TYPES.NO_MODEL
            elif local_model == "LS":
                params.localModel = CCCoreLib.LOCAL_MODEL_TYPES.LS
            elif local_model == "TRI":
                params.localModel = CCCoreLib.LOCAL_MODEL_TYPES.TRI
            elif local_model == "QUADRIC":
                params.localModel = CCCoreLib.LOCAL_MODEL_TYPES.QUADRIC
            print("Set localModel using CCCoreLib.LOCAL_MODEL_TYPES")
        except (ImportError, AttributeError):
            # Alternative 2: Direct integer values (if enum access fails)
            # This should work as a fallback, but the enum is preferred
            local_model_values = {"NO_MODEL": 0, "LS": 1, "TRI": 2, "QUADRIC": 3}
            # Create the enum value manually
            from enum import IntEnum

            class LOCAL_MODEL_TYPES(IntEnum):
                NO_MODEL = 0
                LS = 1
                TRI = 2
                QUADRIC = 3

            params.localModel = LOCAL_MODEL_TYPES[local_model]
            print("Set localModel using fallback IntEnum LOCAL_MODEL_TYPES")
    # Set local model parameters only if not NO_MODEL
    if local_model != "NO_MODEL":
        params.useSphericalSearchForLocalModel = use_spherical_search
        if use_spherical_search:
            params.radiusForLocalModel = sphere_radius
        else:
            params.kNNForLocalModel = knn_points
        params.reuseExistingLocalModels = reuse_existing_local_models

    # Set split distances if requested
    if split_xyz:
        params.setSplitDistances(compared_copy.size())

    # Set CPSet to None (not using closest point set)
    params.CPSet = None

    try:
        # Compute cloud-to-cloud distances
        result_code = cc.DistanceComputationTools.computeCloud2CloudDistances(
            compared_copy, reference_copy, params
        )

        # Check result
        if result_code > 0:  # Success
            return compared_copy, reference_copy, result_code
        else:
            raise RuntimeError(f"Distance computation failed with code: {result_code}")

    except Exception as e:
        raise RuntimeError(f"Error during distance computation: {str(e)}")


def compute_bidirectional_detailed_distances(cloud1, cloud2, **kwargs) -> tuple:
    """
    Compute bidirectional detailed distances between two point clouds.

    Args:
        cloud1: The first point cloud
        cloud2: The second point cloud
        **kwargs: Additional parameters passed to compute_detailed_cloud_distances

    Returns:
        tuple: (cloud1_with_distances_to_cloud2, cloud2_with_distances_to_cloud1, results)
    """

    # Compute cloud1 -> cloud2 distances
    cloud1_result, _, result1 = compute_detailed_cloud_distances(
        cloud1, cloud2, **kwargs
    )

    # Compute cloud2 -> cloud1 distances
    cloud2_result, _, result2 = compute_detailed_cloud_distances(
        cloud2, cloud1, **kwargs
    )

    return cloud1_result, cloud2_result, (result1, result2)


def filter_high_distance(cloud, threshold=0.01) -> list:
    """Filter the point cloud based on a scalar field value.

    Args:
        cloud: The point cloud to filter.
        threshold (float): The threshold value for filtering.
    Returns:
        list: A list of connected components that meet the filtering criteria.
    Raises:
        ValueError: If the cloud is empty or does not have scalar fields.
        TypeError: If the cloud is not a PointCloud instance.
    """
    sf_index = cloud.getNumberOfScalarFields() - 1
    cloud.setCurrentOutScalarField(sf_index)
    filtered = cc.filterBySFValue(threshold, float("inf"), cloud)
    if filtered and filtered.size() > 0:
        filtered.setName(cloud.getName() + "_filtered")
        result = cc.ExtractConnectedComponents(
            clouds=[filtered], octreeLevel=7, randomColors=False
        )
        components = result[1]
        if components:
            return components
    return []


def get_bounding_box(cloud):
    bbox = cloud.getOwnBB()
    bb_min = bbox.minCorner()
    bb_max = bbox.maxCorner()
    return bb_min, bb_max


def clouds_overlap_spatially(cloud1, cloud2, tolerance=0.05):
    bb1_min, bb1_max = get_bounding_box(cloud1)
    bb2_min, bb2_max = get_bounding_box(cloud2)
    overlap_x = not (
        bb1_max[0] + tolerance < bb2_min[0] or bb2_max[0] + tolerance < bb1_min[0]
    )
    overlap_y = not (
        bb1_max[1] + tolerance < bb2_min[1] or bb2_max[1] + tolerance < bb1_min[1]
    )
    overlap_z = not (
        bb1_max[2] + tolerance < bb2_min[2] or bb2_max[2] + tolerance < bb1_min[2]
    )
    return overlap_x and overlap_y and overlap_z


def save_project(entities, project_filepath):
    """
    Save entities (meshes, point clouds, etc.) to a project file

    Args:
        entities (list): List of entities to save
        project_filepath (str): Path where to save the project

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(project_filepath), exist_ok=True)

        # Save entities to a .bin file (CloudCompare's native format)
        result = cc.SaveEntities(entities, project_filepath)

        if result == 0:  # 0 indicates success
            print(f"Successfully saved project to: {project_filepath}")
            return True
        else:
            print(f"Error saving project: {result}")
            return False

    except Exception as e:
        print(f"Error saving project: {e}")
        return False


def find_mesh_by_pgram_job(job_number, mesh_dir=INPUT_MESH_PATH):
    """
    Search for a .ply file in mesh_dir containing 'Pgram_Job_<job_number>' in its filename.
    Returns the full path if found, else None.

    Args:        job_number (int): The job number to search for in the filename.
        mesh_dir (str): The directory to search for the mesh files.

    Returns:
        str: Full path to the mesh file if found, else None.
    """
    pattern = f"Pgram_Job_{job_number}"
    for fname in os.listdir(mesh_dir):
        if fname.endswith(".ply") and pattern in fname:
            return fname.replace(".ply", "")
    return None


def run_presnip_pipeline(json_filepath: str = "input.json") -> None:
    """
    Main entry point for pre-snip processing.
    Loads PLY meshes, samples them, computes cloud-to-cloud distances,
    and saves the distance-tagged clouds for use by auto_snip_script.py.

    Callable from any external Python program:
        import pre_snip_script
        pre_snip_script.run_presnip_pipeline("input.json")
    """
    with open(json_filepath, "r") as f:
        job_data = json.load(f)

    print("Starting pre-snip processing...")

    for job in job_data:
        top_id    = find_mesh_by_pgram_job(job["top"])
        bottom_id = find_mesh_by_pgram_job(job["bottom"])
        print(f"Processing job: Top ID = {top_id}, Bottom ID = {bottom_id}")
        top_mesh    = load_mesh(INPUT_MESH_PATH, top_id)
        bottom_mesh = load_mesh(INPUT_MESH_PATH, bottom_id)
        top_cloud    = sample_mesh(top_mesh)
        bottom_cloud = sample_mesh(bottom_mesh)

        print(f"Computing distances for {top_id} and {bottom_id}...")
        top_with_dist, bottom_with_dist, _ = compute_bidirectional_detailed_distances(
            top_cloud, bottom_cloud
        )
        print(f"Computed distances for {top_id} and {bottom_id}.")

        output_dir = os.path.join(DATA_DIR, top_id)
        os.makedirs(output_dir, exist_ok=True)
        save_cloud(output_dir, top_with_dist,    f"top_with_dist_for_{bottom_id}")
        save_cloud(output_dir, bottom_with_dist, f"bottom_with_dist_for_{top_id}")

    print("Completed pre-snip processing.")


if __name__ == "__main__":
    run_presnip_pipeline(sys.argv[1] if len(sys.argv) > 1 else "input.json")
