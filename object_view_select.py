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
from utils.utils_read import (
    read_extrinsic,
    read_extrinsic_dir,
    read_intrinsic,
    read_depth_map,
    read_bboxes_json,
    load_json,
    reverse_multi2multi_mapping,
    reverse_121_mapping,
    read_annotation_pickles,
    EXCLUDED_OBJECTS,
)
from utils.utils_3d import (
    check_bboxes_visibility,
    check_point_visibility,
    interpolate_bbox_points,
)
from utils.utils_vis import (
    get_9dof_boxes,
    draw_box3d_on_img,
    get_color_map,
    crop_box_from_img,
)


global TRACKED
TRACKED = False


from utils.decorators import mmengine_track_func


@mmengine_track_func
def paint_object_pictures_tracked(bboxes, object_ids, object_types, visible_view_object_dict,
                                  extrinsics_c2w, axis_align_matrix, intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir, output_type="paint", depth_map_paths=None, skip_existing=False):
    paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix,
                          intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir, output_type, depth_map_paths, skip_existing)


def paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir, output_type, depth_map_paths, skip_existing):
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
        image_paths, save_path=blurry_image_ids_path, skip_existing=False
    )
    for image_id in blurry_image_ids:
        if image_id in visible_view_object_dict:
            visible_view_object_dict.pop(image_id)
    visible_object_view_dict = reverse_multi2multi_mapping(
        visible_view_object_dict)
    # NOTE: might have bug for view ids
    view_ids = [os.path.basename(path)[:-4] for path in image_paths]
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)
    color_map = get_color_map()
    image_dir = os.path.dirname(image_paths[0])
    if not depth_map_paths:
        depth_map_paths = [os.path.join(image_dir, view_id + ".png") for view_id in view_ids]
    _paint_object_pictures(bboxes, object_ids, object_types, visible_object_view_dict, extrinsics_c2w, view_ids, intrinsics, depth_intrinsics, depth_map_paths, color_map, image_dir, output_dir,output_type, skip_existing)

@DeprecationWarning
def paint_object_pictures_path(object_json_path, visibility_json_path, extrinsic_dir, axis_align_matrix_path, intrinsic_path, depth_intrinsic_path, depth_map_dir, image_dir, blurry_image_id_path, output_dir, output_type="paint"):
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
    bboxes = get_9dof_boxes(bboxes, "xyz", (0, 0, 192)
                            )  # convert to o3d format
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
    visible_object_view_dict = reverse_multi2multi_mapping(
        visible_view_object_dict)
    extrinsics_c2w, view_ids = read_extrinsic_dir(
        extrinsic_dir)  # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(
        axis_align_matrix_path)  # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)  # c2w
    intrinsic = read_intrinsic(intrinsic_path)  # shape 4, 4
    depth_intrinsic = read_intrinsic(depth_intrinsic_path)  # shape 4, 4
    depth_map_paths = [
        os.path.join(image_dir, view_id + ".png") for view_id in view_ids
    ]
    color_map = get_color_map()
    _paint_object_pictures(bboxes, object_ids, object_types, visible_object_view_dict, extrinsics_c2w, view_ids, intrinsic, depth_intrinsic, depth_map_paths, color_map, image_dir, output_dir, output_type=output_type)


def _paint_object_pictures(bboxes, object_ids, object_types, visible_object_view_dict, extrinsics_c2w, view_ids, intrinsics, depth_intrinsics, depth_map_paths, color_map, image_dir, output_dir,  output_type="paint", skip_existing=True):
    assert output_type in ["paint", "crop"], f"unsupported output type {output_type}"
    assert isinstance(skip_existing, bool), f"expected skip_existing to be a bool, but get {skip_existing}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if skip_existing:
        files_and_folders = os.listdir(output_dir)
        files = [
            f for f in files_and_folders if os.path.isfile(os.path.join(output_dir, f))
        ]
        num_files = len(files)
        valid_objects = [
            obj for obj in object_types if obj not in EXCLUDED_OBJECTS]
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
            file_name = str(object_id).zfill(3) + "_" + \
                object_type + "_placeholder.txt"
            with open(os.path.join(output_dir, file_name), "w") as f:
                f.write("Object not visible in any view.")
            continue
        selected_extrinsics_c2w, selected_intrinsics = [], []
        selected_depth_intrinsics, selected_depth_maps = [], []
        for view_id in visible_views:
            view_index = view_ids.index(view_id) #问题出在view_id是旧一套，但是view_ids是新一套，来自image paths
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
            file_name = str(object_id).zfill(3) + "_" + \
                object_type + "_placeholder.txt"
            with open(os.path.join(output_dir, file_name), "w") as f:
                f.write("Object not visible in any view.")
            continue
        best_view_index = view_ids.index(visible_views[best_view_index])
        best_view_extrinsic_c2w = extrinsics_c2w[best_view_index]
        best_view_intrinsic = intrinsics[best_view_index]
        img_in_path = os.path.join(
            image_dir, view_ids[best_view_index] + ".jpg")
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
                img, bbox, color, label, best_view_extrinsic_c2w, best_view_intrinsic, ignore_outside=False)
        elif output_type == "crop":
            new_img = crop_box_from_img(
                img, bbox, best_view_extrinsic_c2w, best_view_intrinsic
            )
            if new_img is None:
                file_name = (
                    str(object_id).zfill(3) + "_" +
                    object_type + "_placeholder.txt"
                )
                with open(os.path.join(output_dir, file_name), "w") as f:
                    f.write("Object cannot be cropped properly.")
                continue
        cv2.imwrite(img_out_path, new_img)
        if not TRACKED:
            pbar.set_description(
                f"Object {object_id}: {object_type} painted in view {view_ids[best_view_index]} and saved."
            )


def get_visible_objects_dict(object_json_path, extrinsic_dir, axis_align_matrix_path, depth_intrinsic_path, depth_map_dir, output_path, skip_existing=True):
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
    bboxes = get_9dof_boxes(bboxes, "xyz", (0, 0, 192)
                            )  # convert to o3d format
    extrinsics_c2w, view_ids = read_extrinsic_dir(
        extrinsic_dir)  # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(
        axis_align_matrix_path)  # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)  # c2w
    depth_intrinsic = read_intrinsic(depth_intrinsic_path)  # shape 4, 4
    visible_objects_dict = {}
    pbar = tqdm(range(len(extrinsics_c2w)))
    for i in pbar:
        extrinsic_c2w = extrinsics_c2w[i]
        view_id = view_ids[i]
        depth_path = os.path.join(depth_map_dir, view_id + ".png")
        depth_map = read_depth_map(depth_path)  # shape (height, width)
        visibles = check_bboxes_visibility(bboxes, depth_map, depth_intrinsic, np.linalg.inv(
            extrinsic_c2w), corners_only=False, granularity=0.1)
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
        # shape (4, n)
        pts = depth_intrinsic @ np.linalg.inv(extrinsic_c2w) @ points.T
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
    threshold=0,
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
    image_indices = np.argsort(image_ids)
    image_ids = [image_ids[i] for i in image_indices]
    paths = [paths[i] for i in image_indices]
    blurry_ids = []
    vars = []
    variance_dict = {}
    if save_variance_path is None:
        save_variance_dir = os.path.dirname(save_path)
        save_variance_path = os.path.join(
            save_variance_dir, "image_variances.json")
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


def get_blurry_image_ids_dir(image_dir, save_path=None, threshold=0, skip_existing=True, save_variance_path=None):
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
    blurry_ids = get_blurry_image_ids(paths,
                                      save_path=save_path,
                                      threshold=threshold,
                                      skip_existing=False,
                                      save_variance_path=save_variance_path,
                                      )
    return blurry_ids


def is_blurry(image, threshold=0, return_variance=False):
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



mapping_3rscan = load_json('scene_mappings/3rscan_mapping.json') 
mapping_3rscan = reverse_121_mapping(mapping_3rscan) # id to hash
mapping_mp3d = load_json('scene_mappings/mp3d_mapping.json') 
mapping_mp3d = reverse_121_mapping(mapping_mp3d)

def map_view_id(view_id, mode, house_hash=None):
    """
        Args:
            view id: (the numbered view id in anno file). Example: 0165_2_5 (mp3d) or 000137 (3rscan)
            mode: must in 'mp3d' or '3rscan'
        Returns:
            mapped view id (some hash value like b185432bf33645aca813ac2a961b4140_2_5)
    """
    assert mode in ['mp3d', '3rscan'], f"unsupported mode {mode}"
    if mode == 'mp3d':
        assert house_hash is not None
        camera_id = view_id.split('/')[0].split('_')[0]
        angle_id = view_id[-3:]
        mapping_for_cur_scene = load_json(f'scene_mappings/mp3d_rename/{house_hash}.json')
        mapping_for_cur_scene = reverse_121_mapping(mapping_for_cur_scene)
        camera_hash = mapping_for_cur_scene[camera_id]
        return f"{camera_hash}_i{angle_id}" 
    elif mode == '3rscan':
        return f"frame-{view_id}.color"


hashes = []
def map_file_path(image_path):
    if 'mp3d' in image_path:
    # 'posed_images/1mp3d_0015_region0/0165_2_5.jpg'
    # /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/matterport3d/scans/17DRP5sb8fy/matterport_color_images/b185432bf33645aca813ac2a961b4140_i2_3.jpg
    # /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/matterport3d/scans/17DRP5sb8fy/matterport_depth_images/b185432bf33645aca813ac2a961b4140_d2_3.png
        scene_id = image_path.split('/')[-2].split('_region')[0]
        view_id = image_path.split('/')[-1].split('.')[0]
        scene_hash = mapping_mp3d[scene_id]
        hashes.append(scene_hash)
        mapped_view_id = map_view_id(view_id, 'mp3d', scene_hash)
        mapped_view_id_for_depth = mapped_view_id[:-4]+'d'+mapped_view_id[-3:]
        ret_color = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/matterport3d/scans/{scene_hash}/matterport_color_images/{mapped_view_id}.jpg"
        ret_depth = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/matterport3d/scans/{scene_hash}/matterport_depth_images/{mapped_view_id_for_depth}.png"
    # 'posed_images/3rscan0000/000442.jpg'
    # /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/3rscan/raw_data/ffa41874-6f78-2040-85a8-8056ac60c764/sequence/frame-000137.color.jpg
    # /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/3rscan/raw_data/ffa41874-6f78-2040-85a8-8056ac60c764/sequence/frame-000171.depth.pgm
    elif '3rscan' in image_path:
        scene_id = image_path.split('/')[-2]
        view_id = image_path.split('/')[-1].split('.')[0]
        scene_hash = mapping_3rscan[scene_id]
        hashes.append(scene_hash)
        mapped_view_id = map_view_id(view_id, '3rscan')
        mapped_view_id_for_depth = mapped_view_id.replace('color', 'depth')
        ret_color = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/3rscan/raw_data/{scene_hash}/sequence/{mapped_view_id}.jpg"
        ret_depth = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/3rscan/raw_data/{scene_hash}/sequence/{mapped_view_id_for_depth}.pgm"
    elif 'scannet' in image_path:
    # 'scannet/posed_images/scene0701_01/01040.jpg'
    # /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/ScanNet_v2/posed_images/scene0072_00/01040.jpg
    # /mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/ScanNet_v2/posed_images/scene0072_00/01040.png
        scene_id = image_path.split('/')[-2]
        view_id = image_path.split('/')[-1].split('.')[0]
        ret_color = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/ScanNet_v2/posed_images/{scene_id}/{view_id}.jpg"
        ret_depth = f"/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer/ScanNet_v2/posed_images/{scene_id}/{view_id}.png"
    if not os.path.exists(ret_color):
        print(scene_hash)
    return ret_color, ret_depth




if __name__ == "__main__":
    # single_scene_test()
    ########################################################################################
    ## select_for_all_scenes
    ## NOTE: only use the desired pickle files.
    ## No "skipping existing" feature for this code.
    pickle_files = ["embodiedscan_infos_train_full.pkl",
                    "embodiedscan_infos_val_full.pkl",
                    "3rscan_infos_test_MDJH_aligned_full_10_visible.pkl",
                    "matterport3d_infos_test_full_10_visible.pkl"][-1:]
    data_real_dir = "/mnt/hwfile/OpenRobotLab/maoxiaohan/transfer"
    out_real_dir = "/mnt/hwfile/OpenRobotLab/lvruiyuan"
    anno_dict = read_annotation_pickles(pickle_files)
    keys = sorted(list(anno_dict.keys()))
    pbar = tqdm(range(len(keys)))
    inputs = []
    scene_ids = []
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
        _image_paths = anno["image_paths"]
        scene_id = _image_paths[0].split(".")[0].split("/")[-2]
        scene_ids.append(scene_id)
        if '3rscan' in scene_id:
            dataset = '3rscan'
            visible_view_object_dict = {map_view_id(k, '3rscan'):v for k, v in visible_view_object_dict.items()}
        elif 'mp3d' in scene_id:
            dataset = 'matterport3d'
            house_id = _image_paths[0].split('/')[-2].split('_region')[0]
            house_hash = mapping_mp3d[house_id]
            visible_view_object_dict = {map_view_id(k, 'mp3d', house_hash):v for k, v in visible_view_object_dict.items()}
        elif 'scene' in scene_id:
            dataset = 'scannet'
        pbar.set_description(
            "Processing input for scene {}".format(scene_id))
        image_paths, depth_map_paths = [], []
        for path in _image_paths:
            i_path, d_path = map_file_path(path)
            image_paths.append(i_path)
            depth_map_paths.append(d_path)
        blurry_image_ids_path = os.path.join(
            out_real_dir, dataset, scene_id, "blurry_image_ids.json"
        )
        # output_dir = os.path.join(out_real_dir, dataset, scene_id, 'painted_objects')
        # paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir, output_type="paint")
        output_dir = os.path.join(
            out_real_dir, dataset, scene_id, "cropped_objects")
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
            depth_map_paths,
            False # skip existing
        )
        inputs.append(input)
    hashes = set(hashes)
    print(hashes)
    print(scene_ids)
    exit()
    import mmengine
    mmengine.track_parallel_progress(paint_object_pictures_tracked, inputs, nproc=10)
        # paint_object_pictures(*input)
