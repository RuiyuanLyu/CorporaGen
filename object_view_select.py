import numpy as np
import cv2
import os
import json
import shutil
import scipy
from functools import wraps
from tqdm import tqdm
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from utils_read import (
    read_extrinsic,
    read_extrinsic_dir,
    read_intrinsic,
    read_depth_map,
    read_bboxes_json,
    load_json,
    reverse_multi2multi_mapping,
    read_annotation_pickles,
    EXCLUDED_OBJECTS,
)
from utils_3d import (
    check_bboxes_visibility,
    check_point_visibility,
    interpolate_bbox_points,
)
from utils_vis import (
    get_9dof_boxes,
    draw_box3d_on_img,
    get_color_map,
    crop_box_from_img,
)


global TRACKED
TRACKED = False


def mmengine_track_func(func):
    @wraps(func)
    def wrapped_func(args):
        global TRACKED
        TRACKED = True
        result = func(*args)
        return result

    return wrapped_func


@mmengine_track_func
def paint_object_pictures_tracked(
    bboxes,
    object_ids,
    object_types,
    visible_view_object_dict,
    extrinsics_c2w,
    axis_align_matrix,
    intrinsics,
    depth_intrinsics,
    image_paths,
    blurry_image_ids_path,
    output_dir,
    output_type="paint",
):
    paint_object_pictures(
        bboxes,
        object_ids,
        object_types,
        visible_view_object_dict,
        extrinsics_c2w,
        axis_align_matrix,
        intrinsics,
        depth_intrinsics,
        image_paths,
        blurry_image_ids_path,
        output_dir,
        output_type,
    )


def paint_object_pictures(
    bboxes,
    object_ids,
    object_types,
    visible_view_object_dict,
    extrinsics_c2w,
    axis_align_matrix,
    intrinsics,
    depth_intrinsics,
    image_paths,
    blurry_image_ids_path,
    output_dir,
    output_type="paint",
):
    """
    Select the best views for all 3d objects (bboxs) from a set of camera positions (extrinsics) in a scene.
    Then paint the 3d bbox in each view and save the painted images to the output directory.
    Args:
        bboxes: a numpy array of shape (N, 9)
        object_ids: a numpy array of shape (N,) of ints
        object_types: a list of object types (str) of shape (N,)
        visible_view_object_dict: a dictionary of visible objects, where each key is a view index (str) and the value is a list of object ids
        extrinsics_c2w: a list of extrinsic matrices, c2w, shape N, 4, 4
        intrinsics: a list of intrinsic matrices, shape N, 4, 4
        depth_intrinsics: a list of depth intrinsic matrices, shape N, 4, 4
        image_paths: a list of image paths of shape (M,)
        blurry_image_ids_path: path to the json file to contain the blurry image ids
        output_dir: path to the directory to save the painted images to
        output_type: whether "paint" or "crop" the images
    Returns: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bboxes = get_9dof_boxes(np.array(bboxes), mode="zxy", colors=(0, 0, 192))
    blurry_image_ids = get_blurry_image_ids(
        image_paths, save_path=blurry_image_ids_path, skip_existing=True
    )
    for image_id in blurry_image_ids:
        if image_id in visible_view_object_dict:
            visible_view_object_dict.pop(image_id)
    visible_object_view_dict = reverse_multi2multi_mapping(visible_view_object_dict)
    view_ids = [os.path.basename(path).split(".")[0] for path in image_paths]
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)
    color_map = get_color_map()
    image_dir = os.path.dirname(image_paths[0])
    depth_map_paths = [
        os.path.join(image_dir, view_id + ".png") for view_id in view_ids
    ]
    _paint_object_pictures(
        bboxes,
        object_ids,
        object_types,
        visible_object_view_dict,
        extrinsics_c2w,
        view_ids,
        intrinsics,
        depth_intrinsics,
        depth_map_paths,
        color_map,
        image_dir,
        output_dir,
        output_type=output_type,
    )


def paint_object_pictures_path(
    object_json_path,
    visibility_json_path,
    extrinsic_dir,
    axis_align_matrix_path,
    intrinsic_path,
    depth_intrinsic_path,
    depth_map_dir,
    image_dir,
    blurry_image_id_path,
    output_dir,
    output_type="paint",
):
    """
    Select the best views for all 3d objects (bboxs) from a set of camera positions (extrinsics) in a scene.
    Then paint the 3d bbox in each view and save the painted images to the output directory.
    Args:
        object_json_path: path to the json file containing the 3d bboxs of the objects in the scene
        visibility_json_path: path to the json: for each view, a list of object ids that are visible in that view
        extrinsic_dir: path to the directory containing the extrinsic matrices, c2w'
        axis_align_matrix_path: path to the extra extrinsic matrix, w'2w
        intrinsic_path: path to the intrinsic matrix for the scene
        image_dir: path to the directory containing the images for each view
        output_dir: path to the directory to save the painted images to
        output_type: whether "paint" or "crop" the images
    Returns: None
    """
    # Preparings
    bboxes, object_ids, object_types = read_bboxes_json(
        object_json_path, return_id=True, return_type=True
    )
    bboxes = get_9dof_boxes(bboxes, "xyz", (0, 0, 192))  # convert to o3d format
    visible_view_object_dict = get_visible_objects_dict(
        object_json_path,
        extrinsic_dir,
        axis_align_matrix_path,
        depth_intrinsic_path,
        depth_map_dir,
        visibility_json_path,
        skip_existing=True,
    )
    blurry_image_ids = get_blurry_image_ids_dir(
        image_dir, save_path=blurry_image_id_path, skip_existing=True
    )  # image ids are the same as view/extrinsic ids
    for image_id in blurry_image_ids:
        if image_id in visible_view_object_dict:
            visible_view_object_dict.pop(image_id)
    visible_object_view_dict = reverse_multi2multi_mapping(visible_view_object_dict)
    extrinsics_c2w, view_ids = read_extrinsic_dir(extrinsic_dir)  # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(axis_align_matrix_path)  # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)  # c2w
    intrinsic = read_intrinsic(intrinsic_path)  # shape 4, 4
    depth_intrinsic = read_intrinsic(depth_intrinsic_path)  # shape 4, 4
    depth_map_paths = [
        os.path.join(image_dir, view_id + ".png") for view_id in view_ids
    ]
    color_map = get_color_map()
    _paint_object_pictures(
        bboxes,
        object_ids,
        object_types,
        visible_object_view_dict,
        extrinsics_c2w,
        view_ids,
        intrinsic,
        depth_intrinsic,
        depth_map_paths,
        color_map,
        image_dir,
        output_dir,
        output_type=output_type,
    )


def _paint_object_pictures(
    bboxes,
    object_ids,
    object_types,
    visible_object_view_dict,
    extrinsics_c2w,
    view_ids,
    intrinsics,
    depth_intrinsics,
    depth_map_paths,
    color_map,
    image_dir,
    output_dir,
    skip_existing=True,
    output_type="paint",
):
    assert output_type in ["paint", "crop"], "unsupported output type {}".format(
        output_type
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if skip_existing:
        files_and_folders = os.listdir(output_dir)
        files = [
            f for f in files_and_folders if os.path.isfile(os.path.join(output_dir, f))
        ]
        num_files = len(files)
        valid_objects = [obj for obj in object_types if obj not in EXCLUDED_OBJECTS]
        if num_files >= len(valid_objects):
            return
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    depth_maps = [read_depth_map(path) for path in depth_map_paths]
    if len(np.array(intrinsics).shape) == 2:
        intrinsics = np.tile(intrinsics, (len(view_ids), 1, 1))
    if len(np.array(depth_intrinsics).shape) == 2:
        depth_intrinsics = np.tile(depth_intrinsics, (len(view_ids), 1, 1))

    pbar = tqdm(range(len(bboxes))) if not TRACKED else range(len(bboxes))
    for i in pbar:
        bbox, object_id, object_type = bboxes[i], object_ids[i], object_types[i]
        if object_type in EXCLUDED_OBJECTS:
            continue
        visible_views = visible_object_view_dict.get(int(object_id), [])
        if len(visible_views) == 0:
            file_name = str(object_id).zfill(3) + "_" + object_type + "_placeholder.txt"
            with open(os.path.join(output_dir, file_name), "w") as f:
                f.write("Object not visible in any view.")
            continue
        selected_extrinsics_c2w, selected_intrinsics = [], []
        selected_depth_intrinsics, selected_depth_maps = [], []
        for view_id in visible_views:
            view_index = view_ids.index(view_id)
            selected_extrinsics_c2w.append(extrinsics_c2w[view_index])
            selected_intrinsics.append(intrinsics[view_index])
            selected_depth_intrinsics.append(depth_intrinsics[view_index])
            selected_depth_maps.append(depth_maps[view_index])
        selected_extrinsics_c2w = np.array(selected_extrinsics_c2w)
        selected_depth_intrinsics = np.array(selected_depth_intrinsics)
        selected_depth_maps = np.array(selected_depth_maps)
        best_view_index = get_best_view(
            bbox,
            selected_extrinsics_c2w,
            selected_depth_intrinsics,
            selected_depth_maps,
        )
        if best_view_index is None:
            file_name = str(object_id).zfill(3) + "_" + object_type + "_placeholder.txt"
            with open(os.path.join(output_dir, file_name), "w") as f:
                f.write("Object not visible in any view.")
            continue
        best_view_index = view_ids.index(visible_views[best_view_index])
        best_view_extrinsic_c2w = extrinsics_c2w[best_view_index]
        best_view_intrinsic = intrinsics[best_view_index]
        img_in_path = os.path.join(image_dir, view_ids[best_view_index] + ".jpg")
        img_out_path = os.path.join(
            output_dir,
            str(object_id).zfill(3)
            + "_"
            + object_type
            + "_"
            + view_ids[best_view_index]
            + ".jpg",
        )
        img = cv2.imread(img_in_path)
        if img is None:
            # print(f"Image {img_in_path} not found, skipping object {object_id}: {object_type}")
            continue
        color = color_map.get(object_type, (0, 0, 192))
        label = str(object_id) + " " + object_type
        if output_type == "paint":
            new_img, _ = draw_box3d_on_img(
                img,
                bbox,
                color,
                label,
                best_view_extrinsic_c2w,
                best_view_intrinsic,
                ignore_outside=False,
            )
        elif output_type == "crop":
            new_img = crop_box_from_img(
                img, bbox, best_view_extrinsic_c2w, best_view_intrinsic
            )
            if new_img is None:
                file_name = (
                    str(object_id).zfill(3) + "_" + object_type + "_placeholder.txt"
                )
                with open(os.path.join(output_dir, file_name), "w") as f:
                    f.write("Object cannot be cropped properly.")
                continue
        cv2.imwrite(img_out_path, new_img)
        if not TRACKED:
            pbar.set_description(
                f"Object {object_id}: {object_type} painted in view {view_ids[best_view_index]} and saved."
            )


def get_visible_objects_dict(
    object_json_path,
    extrinsic_dir,
    axis_align_matrix_path,
    depth_intrinsic_path,
    depth_map_dir,
    output_path,
    skip_existing=True,
):
    """
    For each camera position (extrinsics), get the visible 3d objects (bboxs).
    Args:
        object_json_path: path to the json file containing the 3d bboxs of the objects in the scene
        extrinsic_dir: path to the directory containing the extrinsic matrices, c2w'
        axis_align_matrix_path: path to the extra extrinsic matrix, w'2w
        depth_intrinsic_path: path to the intrinsic matrix for the depth camera
        depth_map_dir: path to the directory containing the depth maps for each view
        output_path: path to the output json file to save the visible objects
    Returns:
        visible_objects_dict: a dictionary of visible objects, where each key is a view index (str) and the value is a list of object ids
    """
    if os.path.exists(output_path) and skip_existing:
        # print(f"Skipping existing file {output_path}")
        return load_json(output_path)
    bboxes, object_ids, object_types = read_bboxes_json(
        object_json_path, return_id=True, return_type=True
    )
    bboxes = get_9dof_boxes(bboxes, "xyz", (0, 0, 192))  # convert to o3d format
    extrinsics_c2w, view_ids = read_extrinsic_dir(extrinsic_dir)  # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(axis_align_matrix_path)  # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)  # c2w
    depth_intrinsic = read_intrinsic(depth_intrinsic_path)  # shape 4, 4
    visible_objects_dict = {}
    pbar = tqdm(range(len(extrinsics_c2w)))
    for i in pbar:
        extrinsic_c2w = extrinsics_c2w[i]
        view_id = view_ids[i]
        depth_path = os.path.join(depth_map_dir, view_id + ".png")
        depth_map = read_depth_map(depth_path)  # shape (height, width)
        visibles = check_bboxes_visibility(
            bboxes,
            depth_map,
            depth_intrinsic,
            np.linalg.inv(extrinsic_c2w),
            corners_only=False,
            granularity=0.1,
        )
        visible_ids = object_ids[visibles]
        visible_objects_dict[view_id] = visible_ids.tolist()
        pbar.set_description(
            f"View {view_id} has {str(len(visible_ids)).zfill(2)} visible objects"
        )
    with open(output_path, "w") as f:
        json.dump(visible_objects_dict, f, indent=4)
    return visible_objects_dict


def get_best_view(o3d_bbox, extrinsics_c2w, depth_intrinsics, depth_maps):
    """
    Select the best view for an 3d object (bbox) from a set of camera positions (extrinsics)
    Args:
        o3d_bbox: open3d.geometry.OrientedBoundingBox representing the 3d bbox
        extrinsics_c2w: numpy array of shape (n, 4, 4), the extrinsics to select from
        depth_intrinsics: numpy array of shape (n, 4, 4)
        depth_map: numpy array of shape (n, height, width)
    Returns:
        best_view_index: int, the index of the best view in the extrinsics array
    """
    box_center = o3d_bbox.get_center()
    box_center = np.array([box_center[0], box_center[1], box_center[2], 1])
    points = interpolate_bbox_points(o3d_bbox, granularity=0.02)
    points = np.concatenate(
        [points, np.ones_like(points[..., :1])], axis=-1
    )  # shape (n, 4)
    height, width = depth_maps[0].shape
    areas = []
    centerness = []
    for i in range(len(extrinsics_c2w)):
        extrinsic_c2w = extrinsics_c2w[i]
        depth_intrinsic = depth_intrinsics[i]
        depth_map = depth_maps[i]
        pts = depth_intrinsic @ np.linalg.inv(extrinsic_c2w) @ points.T  # shape (4, n)
        xs, ys, zs = pts[0, :], pts[1, :], pts[2, :]
        if zs.min() < 1e-6:
            areas.append(0)
            centerness.append(0)
            continue
        xs, ys = xs / zs, ys / zs
        visible_indices = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys, zs = xs[visible_indices], ys[visible_indices], zs[visible_indices]
        xs, ys = xs.astype(int), ys.astype(int)
        visible_indices = depth_map[ys, xs] > zs
        xs, ys = xs[visible_indices], ys[visible_indices]
        pts = np.stack([xs, ys], axis=-1)
        pts = np.unique(pts, axis=0)
        if len(pts) <= 2:
            areas.append(0)
            centerness.append(0)
            continue
        areas.append(_compute_area(pts))
        # now consider for centerness
        center_pos = (
            depth_intrinsic @ np.linalg.inv(extrinsic_c2w) @ box_center
        )  # shape (4,)
        cx, cy = center_pos[:2] / center_pos[2]
        if (
            width / 4 <= cx
            and cx <= 3 * width / 4
            and height / 4 <= cy
            and cy <= 3 * height / 4
        ):
            centerness.append(1)
        else:
            centerness.append(0)

    areas = np.array(areas)
    centerness = np.array(centerness)
    centered_areas = areas * centerness
    if np.max(centered_areas) == 0:
        if np.max(areas) == 0:
            return None  # object is not visible in any view
        else:
            best_view_index = np.argmax(areas)
            return best_view_index
    best_view_index = np.argmax(centered_areas)
    return best_view_index


def _compute_area(points):
    """
    Computes the area of a set of points.
    """
    try:
        hull = ConvexHull(points)
        area = hull.volume
        return area
    except scipy.spatial.qhull.QhullError as e:
        if "QH6154" in str(e):  # in the same line
            return 0
        if "QH6013" in str(e):  # same x coordinate
            return 0
        else:
            print(points)
            raise e


def _get_convex_hull_points(points):
    """
    get the convex hull of a set of points and sort them in counterclockwise order
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return hull_points


def compute_visible_area(points, image_size):
    """
    Computes the visible area of the convex hull of a set of points (there might be interior points).
    Args:
        points: a numpy array of shape (n,2) representing the points in the plane
        image_size: tuple of (width, height) of the image
    Returns: float, the area of the area of the convex hull
    """
    image_coords = np.array(
        [[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]]
    )
    poly1 = Polygon(_get_convex_hull_points(points))
    poly2 = Polygon(image_coords)
    intersection = poly1.intersection(poly2)
    area = intersection.area
    if area > image_size[0] * image_size[1] / 2:
        return 0  # discard those are too large
    return area


def get_blurry_image_ids(
    image_paths,
    save_path=None,
    threshold=150,
    skip_existing=False,
    save_variance_path=None,
):
    """
    Returns a list of image ids that are blurry.
    Args:
        image_paths: a list of image paths
        save_path: path to the output json file to save the blurry image ids
        threshold: the threshold for the variance of the laplacian to consider an image blurry
        save_variance_path: path to the output json file to save the variance of each image
    Returns:
        blurry_ids: a list of image ids that are blurry
    """
    if os.path.exists(save_path) and skip_existing:
        # print(f"Skipping existing file {save_path}")
        return load_json(save_path)
    image_ids = []
    paths = image_paths
    for image_path in image_paths:
        image_id = os.path.basename(image_path)[:-4]
        image_ids.append(image_id)
    image_indices = np.argsort([int(i) for i in image_ids])
    image_ids = [image_ids[i] for i in image_indices]
    paths = [paths[i] for i in image_indices]
    blurry_ids = []
    vars = []
    variance_dict = {}
    if save_variance_path is None:
        save_variance_dir = os.path.dirname(save_path)
        save_variance_path = os.path.join(save_variance_dir, "image_variances.json")
    if os.path.exists(save_variance_path):
        variance_dict = load_json(save_variance_path)
    pbar = tqdm(range(len(paths)))
    for i in pbar:
        image_path = paths[i]
        image_id = image_ids[i]
        pbar.set_description(f"Checking {image_id}")
        if image_id in variance_dict:
            variance = variance_dict[image_id]
            vars.append(variance)
            if variance < threshold:
                blurry_ids.append(image_id)
            continue
        image = cv2.imread(image_path)
        blurry, variance = is_blurry(image, threshold, return_variance=True)
        vars.append(variance)
        if blurry:
            blurry_ids.append(image_id)
    not_blurry_indices = get_local_maxima_indices(
        vars
    )  # only applies for consecutive image streams
    for i in not_blurry_indices:
        if image_ids[i] in blurry_ids:
            blurry_ids.remove(image_ids[i])
    print(f"Found {len(blurry_ids)} blurry images out of {len(image_ids)}")
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(blurry_ids, f, indent=4)
    variance_dict = {}
    for i in range(len(image_ids)):
        variance_dict[image_ids[i]] = vars[i]
    with open(save_variance_path, "w") as f:
        json.dump(variance_dict, f, indent=4)
    return blurry_ids


def get_blurry_image_ids_dir(
    image_dir,
    save_path=None,
    threshold=150,
    skip_existing=True,
    save_variance_path=None,
):
    """
    Returns a list of image ids that are blurry.
    Args:
        image_dir: path to the directory containing the images
        save_path: path to the output json file to save the blurry image ids
        threshold: the threshold for the variance of the laplacian to consider an image blurry
        save_variance_path: path to the output json file to save the variance of each image
    Returns:
        blurry_ids: a list of image ids that are blurry
    """
    if os.path.exists(save_path) and skip_existing:
        # print(f"Skipping existing file {save_path}")
        return load_json(save_path)
    paths = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".jpg"):
            image_id = file_name[:-4]
            image_path = os.path.join(image_dir, file_name)
            paths.append(image_path)
    blurry_ids = get_blurry_image_ids(
        paths,
        save_path=save_path,
        threshold=threshold,
        skip_existing=False,
        save_variance_path=save_variance_path,
    )
    return blurry_ids


def is_blurry(image, threshold=100, return_variance=False):
    """
    Returns True if the image is blurry, False otherwise.
    The lower the variance of the laplacian, the more blurry the image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    if return_variance:
        return variance < threshold, variance
    return variance < threshold


def get_local_maxima_indices(data, window_size=3):
    """
    Returns the local maxima indices of a 1D array.
    """
    maxima_indices = []
    if len(data) < window_size:
        return maxima_indices
    for i in range(len(data)):
        left_index = max(i - window_size, 0)
        right_index = min(i + window_size, len(data) - 1)
        window = data[left_index:right_index]
        if data[i] == np.max(window) and data[i] > 0:
            maxima_indices.append(i)
    return maxima_indices


def single_scene_test_local():
    object_json_path = f"./example_data/label/main_MDJH13.json"  # need to exist
    visibility_json_path = f"./example_data/anno_lang/visible_objects.json"  # can be generated by this script
    extrinsic_dir = "./example_data/posed_images"  # need to exist
    axis_align_matrix_path = "./example_data/label/rot_matrix.npy"  # need to exist
    intrinsic_path = f"./example_data/posed_images/intrinsic.txt"  # need to exist
    image_dir = "./example_data/posed_images"  # need to exist
    blurry_image_id_path = "./example_data/anno_lang/blurry_image_ids.json"  # can be generated by this script
    save_variance_path = "./example_data/anno_lang/image_variances.json"  # can be generated by this script
    output_dir = "./example_data/anno_lang/painted_images"  # need to exist
    depth_intrinsic_path = (
        f"./example_data/posed_images/depth_intrinsic.txt"  # need to exist
    )
    depth_map_dir = "./example_data/posed_images"  # need to exist
    get_blurry_image_ids_dir(
        image_dir,
        save_path=blurry_image_id_path,
        save_variance_path=save_variance_path,
        skip_existing=False,
    )
    get_visible_objects_dict(
        object_json_path,
        extrinsic_dir,
        axis_align_matrix_path,
        depth_intrinsic_path,
        depth_map_dir,
        visibility_json_path,
        skip_existing=False,
    )
    paint_object_pictures_path(
        object_json_path,
        visibility_json_path,
        extrinsic_dir,
        axis_align_matrix_path,
        intrinsic_path,
        depth_intrinsic_path,
        depth_map_dir,
        image_dir,
        blurry_image_id_path,
        output_dir,
    )


def single_scene_test_by_pickle():
    pickle_file_val = "./example_data/embodiedscan_infos_val_full.pkl"
    pickle_file_train = "./example_data/embodiedscan_infos_train_full.pkl"
    anno_dict = read_annotation_pickles([pickle_file_val, pickle_file_train])
    keys = sorted(list(anno_dict.keys()))
    for key in keys:
        anno = anno_dict[key]
        bboxes = anno["bboxes"]
        object_ids = anno["object_ids"]
        object_types = anno["object_types"]
        visible_view_object_dict = anno["visible_view_object_dict"]
        extrinsics_c2w = anno["extrinsics_c2w"]
        axis_align_matrix = anno["axis_align_matrix"]
        intrinsics = anno["intrinsics"]
        depth_intrinsics = anno["depth_intrinsics"]
        image_paths = anno["image_paths"]

        dataset, _, scene_id, _ = image_paths[0].split(".")[0].split("/")
        if dataset != "scannet":
            continue
        if scene_id != "scene0000_00":
            continue
        print(f"Processing {dataset} {scene_id}")
        real_image_dir = "./example_data/posed_images"
        real_image_paths = []
        for image_path in image_paths:
            image_id = os.path.basename(image_path)[:-4]
            real_image_paths.append(os.path.join(real_image_dir, image_id + ".jpg"))
        print(f"Real image path example: {real_image_paths[0]}")
        blurry_image_ids_path = "./example_data/anno_lang/blurry_image_ids.json"
        # output_dir = './example_data/anno_lang/painted_images'
        # paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, depth_intrinsics, real_image_paths, blurry_image_ids_path, output_dir, output_type="paint")
        output_dir = "./example_data/anno_lang/cropped_images"
        paint_object_pictures(
            bboxes,
            object_ids,
            object_types,
            visible_view_object_dict,
            extrinsics_c2w,
            axis_align_matrix,
            intrinsics,
            depth_intrinsics,
            real_image_paths,
            blurry_image_ids_path,
            output_dir,
            output_type="crop",
        )


def select_for_all_scenes_single_thread():
    pickle_file_val = "/mnt/petrelfs/share_data/wangtai/data/full_10_visible/embodiedscan_infos_val_full.pkl"
    pickle_file_train = "/mnt/petrelfs/share_data/wangtai/data/full_10_visible/embodiedscan_infos_train_full.pkl"
    data_real_dir = "/mnt/petrelfs/share_data/maoxiaohan"
    out_real_dir = "/mnt/petrelfs/share_data/lvruiyuan"
    anno_dict = read_annotation_pickles([pickle_file_val, pickle_file_train])
    keys = sorted(list(anno_dict.keys()))
    pbar = tqdm(range(len(keys)))
    for key_index in pbar:
        key = keys[key_index]
        anno = anno_dict[key]
        bboxes = anno["bboxes"]
        object_ids = anno["object_ids"]
        object_types = anno["object_types"]
        visible_view_object_dict = anno["visible_view_object_dict"]
        extrinsics_c2w = anno["extrinsics_c2w"]
        axis_align_matrix = anno["axis_align_matrix"]
        intrinsics = anno["intrinsics"]
        depth_intrinsics = anno["depth_intrinsics"]
        image_paths = anno["image_paths"]

        dataset, _, scene_id, _ = image_paths[0].split(".")[0].split("/")
        pbar.set_description("Painting for {} scene {}".format(dataset, scene_id))
        image_paths = [
            os.path.join(
                data_real_dir,
                path.replace("matterport3d", "matterport3d/matterport3d").replace(
                    "scannet", "ScanNet_v2"
                ),
            )
            for path in image_paths
        ]  # dirty implementation. The real data is not arranged properly.
        blurry_image_ids_path = os.path.join(
            out_real_dir, dataset, scene_id, "blurry_image_ids.json"
        )
        # output_dir = os.path.join(out_real_dir, dataset, scene_id, 'painted_objects')
        # paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir, output_type="paint")
        output_dir = os.path.join(out_real_dir, dataset, scene_id, "cropped_objects")
        paint_object_pictures(
            bboxes,
            object_ids,
            object_types,
            visible_view_object_dict,
            extrinsics_c2w,
            axis_align_matrix,
            intrinsics,
            depth_intrinsics,
            image_paths,
            blurry_image_ids_path,
            output_dir,
            output_type="crop",
        )


def select_for_all_scenes_multi_threads():
    pickle_file_val = "/mnt/petrelfs/share_data/wangtai/data/full_10_visible/embodiedscan_infos_val_full.pkl"
    pickle_file_train = "/mnt/petrelfs/share_data/wangtai/data/full_10_visible/embodiedscan_infos_train_full.pkl"
    data_real_dir = "/mnt/petrelfs/share_data/maoxiaohan"
    out_real_dir = "/mnt/petrelfs/share_data/lvruiyuan"
    anno_dict = read_annotation_pickles([pickle_file_val, pickle_file_train])
    keys = sorted(list(anno_dict.keys()))
    inputs = []
    for key_index in range(len(keys)):
        key = keys[key_index]
        anno = anno_dict[key]
        bboxes = anno["bboxes"]
        object_ids = anno["object_ids"]
        object_types = anno["object_types"]
        visible_view_object_dict = anno["visible_view_object_dict"]
        extrinsics_c2w = anno["extrinsics_c2w"]
        axis_align_matrix = anno["axis_align_matrix"]
        intrinsics = anno["intrinsics"]
        depth_intrinsics = anno["depth_intrinsics"]
        image_paths = anno["image_paths"]

        dataset, _, scene_id, _ = image_paths[0].split(".")[0].split("/")
        image_paths = [
            os.path.join(
                data_real_dir,
                path.replace("matterport3d", "matterport3d/matterport3d").replace(
                    "scannet", "ScanNet_v2"
                ),
            )
            for path in image_paths
        ]  # dirty implementation. The real data is not arranged properly.
        blurry_image_ids_path = os.path.join(
            out_real_dir, dataset, scene_id, "blurry_image_ids.json"
        )
        # output_dir = os.path.join(out_real_dir, dataset, scene_id, 'painted_objects')
        # paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir, output_type="paint")
        output_dir = os.path.join(out_real_dir, dataset, scene_id, "cropped_objects")

        input = (
            bboxes,
            object_ids,
            object_types,
            visible_view_object_dict,
            extrinsics_c2w,
            axis_align_matrix,
            intrinsics,
            depth_intrinsics,
            image_paths,
            blurry_image_ids_path,
            output_dir,
            "crop",
        )
        inputs.append(input)
    import mmengine

    mmengine.utils.track_parallel_progress(
        func=paint_object_pictures_tracked, tasks=inputs, nproc=8
    )


if __name__ == "__main__":
    # single_scene_test()
    select_for_all_scenes_multi_threads()
