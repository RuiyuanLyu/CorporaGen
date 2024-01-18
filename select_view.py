import numpy as np
import cv2
import os
import json
import tqdm
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from utils_read import read_extrinsic, read_extrinsic_dir, read_intrinsic, read_bboxes_json
from utils_3d import check_bboxes_visibility
from visualization import get_9dof_boxes, draw_box3d_on_img, get_color_map


def paint_object_pictures(object_json_path, extrinsic_dir, axis_align_matrix_path, intrinsic_path, image_dir, output_dir):
    """
        Select the best views for all 3d objects (bboxs) from a set of camera positions (extrinsics) in a scene.
        Then paint the 3d bbox in each view and save the painted images to the output directory.
        Args:
            object_json_path: path to the json file containing the 3d bboxs of the objects in the scene
            extrinsic_dir: path to the directory containing the extrinsic matrices, c2w' 
            axis_align_matrix_path: path to the extra extrinsic matrix, w'2w
            intrinsic_path: path to the intrinsic matrix for the scene
            image_dir: path to the directory containing the images for each view
            output_dir: path to the directory to save the painted images to
        Returns: None
    """
    bboxes, ids, types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    bboxes = get_9dof_boxes(bboxes, 'xyz', (0, 0, 192)) # convert to o3d format
    extrinsics_c2w, extrinsic_paths = read_extrinsic_dir(extrinsic_dir) # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(axis_align_matrix_path) # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w) # c2w
    intrinsic = read_intrinsic(intrinsic_path) # shape 4, 4
    color_map = get_color_map()
    sample_img_path = os.path.join(image_dir, extrinsic_paths[0].split('/')[-1][:-4] + '.jpg')
    height, width = cv2.imread(sample_img_path).shape[:2]
    image_size = (width, height)
    for bbox, id, type in zip(bboxes, ids, types):
        best_view, best_view_index = get_best_view(bbox, extrinsics_c2w, intrinsic, image_size)
        img_in_path = os.path.join(image_dir, extrinsic_paths[best_view_index].split('/')[-1][:-4] + '.jpg')
        img_out_path = os.path.join(output_dir, str(id).zfill(3) + '_' + type + '.jpg')
        img = cv2.imread(img_in_path)
        if img is None:
            print(f"best_view_index: {best_view_index}, extrinsic_path: {extrinsic_paths[best_view_index]}")
            print(f"Image {img_in_path} not found, skipping object {id}: {type}")
            continue
        color = color_map.get(type, (0, 0, 192))
        label = str(id) + ' ' + type
        painted_img, _ = draw_box3d_on_img(img, bbox, color, label, best_view, intrinsic)
        cv2.imwrite(img_out_path, painted_img)
        print(f"painted image {img_out_path} for object {id}: {type}")

def get_visible_objects_dict(object_json_path, extrinsic_dir, axis_align_matrix_path, depth_intrinsic_path, depth_map_dir, output_path):
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
    bboxes, ids, types = read_bboxes_json(object_json_path, return_id=True, return_type=True)
    bboxes = get_9dof_boxes(bboxes, 'xyz', (0, 0, 192)) # convert to o3d format
    extrinsics_c2w, extrinsic_paths = read_extrinsic_dir(extrinsic_dir) # c2w', shape N, 4, 4
    axis_align_matrix = read_extrinsic(axis_align_matrix_path) # w'2w, shape 4, 4
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w) # c2w
    depth_intrinsic = read_intrinsic(depth_intrinsic_path) # shape 4, 4
    visible_objects_dict = {}
    for i, extrinsic_c2w in enumerate(extrinsics_c2w):
        view_index = extrinsic_paths[i].split('/')[-1][:-4]
        depth_path = os.path.join(depth_map_dir, view_index + '.png')
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0 # shape (height, width)
        visibles = check_bboxes_visibility(bboxes, depth_map, depth_intrinsic, np.linalg.inv(extrinsic_c2w)) # bool array of shape N
        visible_ids = ids[visibles]
        visible_objects_dict[view_index] = visible_ids.tolist()
        # print(f"View {view_index} has {len(visible_ids)} visible objects")
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


if __name__ == '__main__':
    object_json_path = f"./example_data/label/main_MDJH13.json"
    extrinsic_dir = "./example_data/posed_images"
    axis_align_matrix_path = "./example_data/label/rot_matrix.npy"
    intrinsic_path = f"./example_data/posed_images/intrinsic.txt"
    image_dir = "./example_data/posed_images"
    output_dir = "./example_data/anno_lang/painted_images"
    # paint_object_pictures(object_json_path, extrinsic_dir, axis_align_matrix_path, intrinsic_path, image_dir, output_dir)
    depth_intrinsic_path = f"./example_data/posed_images/depth_intrinsic.txt"
    depth_map_dir = "./example_data/posed_images"
    output_path = f"./example_data/anno_lang/visible_objects.json"
    get_visible_objects_dict(object_json_path, extrinsic_dir, axis_align_matrix_path, depth_intrinsic_path, depth_map_dir, output_path)