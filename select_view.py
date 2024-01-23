import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from utils_read import read_extrinsic, read_extrinsic_dir, read_intrinsic, read_bboxes_json, load_json, reverse_multi2multi_mapping
from utils_3d import check_bboxes_visibility
from visualization import get_9dof_boxes, draw_box3d_on_img, get_color_map


def paint_object_pictures(object_json_path, visibility_json_path, extrinsic_dir, axis_align_matrix_path, intrinsic_path, depth_intrinsic_path, depth_map_dir, image_dir, blurry_image_id_path, output_dir):
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
        Returns: None
    """
    bboxes, object_ids, object_types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    bboxes = get_9dof_boxes(bboxes, 'xyz', (0, 0, 192)) # convert to o3d format
    visible_view_object_dict = get_visible_objects_dict(object_json_path, extrinsic_dir, axis_align_matrix_path, depth_intrinsic_path, depth_map_dir, visibility_json_path, skip_existing=True)
    blurry_image_ids = get_blurry_image_ids(image_dir, save_path=blurry_image_id_path, threshold=200, skip_existing=True) # image ids are the same as view/extrinsic ids
    for image_id in blurry_image_ids:
        if image_id in visible_view_object_dict:
            visible_view_object_dict.pop(image_id)
    visible_object_view_dict = reverse_multi2multi_mapping(visible_view_object_dict)
    extrinsics_c2w, extrinsic_ids = read_extrinsic_dir(extrinsic_dir) # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(axis_align_matrix_path) # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w) # c2w
    intrinsic = read_intrinsic(intrinsic_path) # shape 4, 4
    color_map = get_color_map()
    sample_img_path = os.path.join(image_dir, extrinsic_ids[0] + '.jpg')
    height, width = cv2.imread(sample_img_path).shape[:2]
    image_size = (width, height)
    pbar = tqdm(range(len(bboxes)))
    for i in pbar:
        bbox, object_id, object_type = bboxes[i], object_ids[i], object_types[i]
        visible_views = visible_object_view_dict.get(int(object_id), [])
        selected_extrinsics_c2w = []
        for view_id in visible_views:
            extrinsic_index = extrinsic_ids.index(view_id)
            extrinsic_c2w = extrinsics_c2w[extrinsic_index]
            selected_extrinsics_c2w.append(extrinsic_c2w)
        if len(selected_extrinsics_c2w) == 0:
            print(f"Object {object_id}: {object_type} not visible in any view, skipping")
            # create a placeholder txt
            file_name = str(object_id).zfill(3) + '_' + object_type + '_placeholder.txt'
            with open(os.path.join(output_dir, file_name), 'w') as f:
                f.write('Object not visible in any view')
            continue
        selected_extrinsics_c2w = np.array(selected_extrinsics_c2w)
        best_view, best_view_index = get_best_view(bbox, selected_extrinsics_c2w, intrinsic, image_size)
        best_view_index = extrinsic_ids.index(visible_views[best_view_index])
        img_in_path = os.path.join(image_dir, extrinsic_ids[best_view_index] + '.jpg')
        img_out_path = os.path.join(output_dir, str(object_id).zfill(3) + '_' + object_type + '_' + extrinsic_ids[best_view_index] + '.jpg')
        img = cv2.imread(img_in_path)
        if img is None:
            print(f"best_view_index: {best_view_index}, extrinsic_id: {extrinsic_ids[best_view_index]}")
            print(f"Image {img_in_path} not found, skipping object {object_id}: {object_type}")
            continue
        color = color_map.get(object_type, (0, 0, 192))
        label = str(object_id) + ' ' + object_type
        painted_img, _ = draw_box3d_on_img(img, bbox, color, label, best_view, intrinsic, ignore_outside=False)
        cv2.imwrite(img_out_path, painted_img)
        pbar.set_description(f"Object {object_id}: {object_type} painted in view {extrinsic_ids[best_view_index]} and saved.")

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
        print(f"Skipping existing file {output_path}")
        return load_json(output_path)
    bboxes, object_ids, object_types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    bboxes = get_9dof_boxes(bboxes, 'xyz', (0, 0, 192)) # convert to o3d format
    extrinsics_c2w, extrinsic_ids = read_extrinsic_dir(extrinsic_dir) # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(axis_align_matrix_path) # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w) # c2w
    depth_intrinsic = read_intrinsic(depth_intrinsic_path) # shape 4, 4
    visible_objects_dict = {}
    pbar = tqdm(range(len(extrinsics_c2w)))
    for i in pbar:
        extrinsic_c2w = extrinsics_c2w[i]
        view_index = extrinsic_ids[i]
        depth_path = os.path.join(depth_map_dir, view_index + '.png')
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0 # shape (height, width)
        visibles = check_bboxes_visibility(bboxes, depth_map, depth_intrinsic, np.linalg.inv(extrinsic_c2w), corners_only=False, granularity=0.1)
        visible_ids = object_ids[visibles]
        visible_objects_dict[view_index] = visible_ids.tolist()
        pbar.set_description(f"View {view_index} has {str(len(visible_ids)).zfill(2)} visible objects")
    with open(output_path, 'w') as f:
        json.dump(visible_objects_dict, f, indent=4)
    return visible_objects_dict

def get_best_view(o3d_bbox, extrinsics_c2w, intrinsic, image_size):
    """
        Select the best view for an 3d object (bbox) from a set of camera positions (extrinsics)
        Args:
            o3d_bbox: open3d.geometry.OrientedBoundingBox representing the 3d bbox
            extrinsics_c2w: numpy array of shape (n, 4, 4) representing the extrinsics to select from
            intrinsic: numpy array of shape (4, 4) representing the intrinsics matrix
            image_size: tuple of (width, height) of the image
        Returns:
            best_view: numpy array of shape (4, 4), the extrinsic matrix of the best view
            best_view_index: int, the index of the best view in the extrinsics array
    """
    corners = np.array(o3d_bbox.get_box_points()) # shape (8, 3)
    corners = np.concatenate([corners, np.ones((8, 1))], axis=1) # shape (8, 4)
    areas = []
    num_insides = []
    for extrinsic in extrinsics_c2w:
        projected_corners = intrinsic @ np.linalg.inv(extrinsic) @ corners.T # shape (4, 8)
        if projected_corners[2, :].min() < 0: # check if the object is behind the camera
            num_insides.append(0)
            continue
        projected_corners = (projected_corners[:2, :] / projected_corners[2, :]).T # shape (8, 2)
        num_inside = np.sum(is_inside_2d_box(projected_corners, image_size))
        num_insides.append(num_inside)
    num_insides = np.array(num_insides)
    good_indices = np.where(num_insides==np.max(num_insides))[0]
    for i in good_indices:
        extrinsic = extrinsics_c2w[i]
        projected_corners = intrinsic @ np.linalg.inv(extrinsic) @ corners.T # shape (4, 8)
        projected_corners = (projected_corners[:2, :] / projected_corners[2, :]).T # shape (8, 2)
        areas.append(compute_visible_area(projected_corners, image_size))
    areas = np.array(areas)
    _best_view_index = np.argmax(areas)
    best_view_index = good_indices[_best_view_index]
    best_view = extrinsics_c2w[best_view_index]
    best_area = areas[_best_view_index]
    # print(f"Best view index: {best_view_index}, best area: {best_area}")
    # if best_area > 0:
    #     best_projection = intrinsic @ np.linalg.inv(best_view) @ corners.T # shape (4, 8)
    #     best_projection = (best_projection[:2, :] / best_projection[2, :]).T # shape (8, 2)
    #     print(best_projection)
    return best_view, best_view_index

def is_inside_2d_box(points, box_size):
    """
        Check if a set of points are inside a 2d box
        Args:
            points: a numpy array of shape (n,2) representing the points in the plane
            box_size: a tuple of (width, height) of the box
        Returns: a boolean array of shape (n,) indicating if each point is inside the box
    """
    width, height = box_size
    x_min, y_min = 0, 0
    x_max, y_max = width - 1, height - 1
    inside = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    return inside

def get_convex_hull_points(points):
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
    image_coords = np.array([[0, 0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]])
    poly1 = Polygon(get_convex_hull_points(points))
    poly2 = Polygon(image_coords)
    intersection = poly1.intersection(poly2)
    return intersection.area

def get_blurry_image_ids(image_dir, save_path=None, threshold=200, skip_existing=True, save_variance_path=None):
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
        print(f"Skipping existing file {save_path}")
        return load_json(save_path)
    blurry_ids = []
    image_ids = []
    vars = []
    paths = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg'):
            image_id = file_name[:-4]
            image_path = os.path.join(image_dir, file_name)
            paths.append(image_path)
            image_ids.append(image_id)
    # sort the images by id
    image_indices = np.argsort([int(i) for i in image_ids])
    image_ids = [image_ids[i] for i in image_indices]
    paths = [paths[i] for i in image_indices]
    variance_dict = {}
    if save_variance_path is not None:
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
    not_blurry_indices = get_local_maxima_indices(vars) # only applies for consecutive image streams
    for i in not_blurry_indices:
        if image_ids[i] in blurry_ids:
            blurry_ids.remove(image_ids[i])
    print(f"Found {len(blurry_ids)} blurry images out of {len(image_ids)}")
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(blurry_ids, f, indent=4)
    if save_variance_path is not None:
        variance_dict = {}
        for i in range(len(image_ids)):
            variance_dict[image_ids[i]] = vars[i]
        with open(save_variance_path, 'w') as f:
            json.dump(variance_dict, f, indent=4)
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
    laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    if return_variance:
        return variance < threshold, variance
    return variance < threshold

def get_local_maxima_indices(data, window_size=3):
    """
        Returns the local maxima indices of a 1D array.
    """
    maxima_indices = []
    for i in range(len(data)):
        left_index = max(i - window_size, 0)
        right_index = min(i + window_size, len(data) - 1)
        window = data[left_index:right_index]
        if data[i] == np.max(window) and data[i] > 0:
            maxima_indices.append(i)
    return maxima_indices


if __name__ == '__main__':
    object_json_path = f"./example_data/label/main_MDJH13.json" # need to exist
    visibility_json_path = f"./example_data/anno_lang/visible_objects.json" # can be generated by this script
    extrinsic_dir = "./example_data/posed_images" # need to exist
    axis_align_matrix_path = "./example_data/label/rot_matrix.npy" # need to exist
    intrinsic_path = f"./example_data/posed_images/intrinsic.txt" # need to exist
    image_dir = "./example_data/posed_images" # need to exist
    blurry_image_id_path = "./example_data/anno_lang/blurry_image_ids.json" # can be generated by this script
    save_variance_path = "./example_data/anno_lang/image_variances.json" # can be generated by this script
    output_dir = "./example_data/anno_lang/painted_images" # need to exist
    depth_intrinsic_path = f"./example_data/posed_images/depth_intrinsic.txt" # need to exist
    depth_map_dir = "./example_data/posed_images" # need to exist
    get_blurry_image_ids(image_dir, save_path=blurry_image_id_path, threshold=200, save_variance_path=save_variance_path, skip_existing=False)
    get_visible_objects_dict(object_json_path, extrinsic_dir, axis_align_matrix_path, depth_intrinsic_path, depth_map_dir, visibility_json_path, skip_existing=False)
    paint_object_pictures(object_json_path, visibility_json_path, extrinsic_dir, axis_align_matrix_path, intrinsic_path, depth_intrinsic_path, depth_map_dir, image_dir, blurry_image_id_path, output_dir) 
