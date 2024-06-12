import numpy as np
import os
import cv2
import copy
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import ConvexHull
import scipy
from shapely.geometry import Polygon
from utils.utils_read import read_extrinsic, read_extrinsic_dir, read_intrinsic, read_depth_map, read_bboxes_json, load_json, reverse_multi2multi_mapping, read_annotation_pickles, EXCLUDED_OBJECTS
from utils.utils_3d import check_bboxes_visibility, check_point_visibility, interpolate_bbox_points
import shutil
import json
from region_matching import get_data,process_data
from utils.utils_vis import get_o3d_obb, draw_box3d_on_img, get_color_map, crop_box_from_img
import matplotlib.pyplot as plt
from object_view_select import get_local_maxima_indices, is_blurry, get_blurry_image_ids, _compute_area



def paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix,
                          intrinsics, depth_intrinsics, image_paths, blurry_image_ids_path, output_dir,
                          output_type="paint"):
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

    bboxes = get_o3d_obb(np.array(bboxes), mode='zxy', colors=(0, 0, 192))
    blurry_image_ids = get_blurry_image_ids(image_paths, save_path=blurry_image_ids_path, skip_existing=True)
    for image_id in blurry_image_ids:
        if image_id in visible_view_object_dict:
            visible_view_object_dict.pop(image_id)
    visible_object_view_dict = reverse_multi2multi_mapping(visible_view_object_dict)

    view_ids = [os.path.basename(path).split('.')[0] for path in image_paths]
    extrinsics_c2w = np.matmul(axis_align_matrix, extrinsics_c2w)
    color_map = get_color_map()
    image_dir = os.path.dirname(image_paths[0])
    depth_map_paths = [os.path.join(image_dir, view_id + '.png') for view_id in view_ids]
    _paint_object_pictures(bboxes, object_ids, object_types, visible_object_view_dict, extrinsics_c2w, view_ids,
                           intrinsics, depth_intrinsics, depth_map_paths, color_map, image_dir, output_dir,
                           output_type=output_type)


def get_best_view(o3d_bbox, extrinsics_c2w, depth_intrinsics, depth_maps,show=False):
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
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)  # shape (n, 4)
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
        center_pos = depth_intrinsic @ np.linalg.inv(extrinsic_c2w) @ box_center  # shape (4,)
        cx, cy = center_pos[:2] / center_pos[2]

        n = 4
        if width / n <= cx and cx <= (n-1) * width / n and height / n <= cy and cy <= (n-1) * height / n:
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


def _paint_object_pictures(bboxes, object_ids, object_types, visible_object_view_dict, extrinsics_c2w, view_ids,
                           intrinsics, depth_intrinsics, depth_map_paths, color_map, image_dir, output_dir,
                           skip_existing=True, output_type="paint"):
    assert output_type in ["paint", "crop"], "unsupported output type {}".format(output_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if skip_existing:
        files_and_folders = os.listdir(output_dir)
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(output_dir, f))]
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

    pbar = tqdm(range(len(bboxes)))
    for i in pbar:

        bbox, object_id, object_type = bboxes[i], object_ids[i], object_types[i]

        if object_type in EXCLUDED_OBJECTS:
            continue
        visible_views = visible_object_view_dict.get(int(object_id), [])

        if len(visible_views) == 0:
            file_name = str(object_id).zfill(3) + '_' + object_type + '_placeholder.txt'
            with open(os.path.join(output_dir, file_name), 'w') as f:
                f.write('Object not visible in any view.')
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

        best_view_index = get_best_view(bbox, selected_extrinsics_c2w, selected_depth_intrinsics, selected_depth_maps,show=(object_id==150))
        if best_view_index is None:
            file_name = str(object_id).zfill(3) + '_' + object_type + '_placeholder.txt'
            with open(os.path.join(output_dir, file_name), 'w') as f:
                f.write('Object not visible in any view.')
            continue
        best_view_index = view_ids.index(visible_views[best_view_index])
        best_view_extrinsic_c2w = extrinsics_c2w[best_view_index]
        best_view_intrinsic = intrinsics[best_view_index]
        img_in_path = os.path.join(image_dir, view_ids[best_view_index] + '.jpg')
        img_out_path = os.path.join(output_dir, str(object_id).zfill(3) + '_' + object_type + '_' + view_ids[
            best_view_index] + '.jpg')
        img = cv2.imread(img_in_path)
        if img is None:
            # print(f"Image {img_in_path} not found, skipping object {object_id}: {object_type}")
            continue
        color = color_map.get(object_type, (0, 0, 192))
        label = str(object_id) + ' ' + object_type
        if output_type == "paint":
            new_img, _ = draw_box3d_on_img(img, bbox, color, label, best_view_extrinsic_c2w, best_view_intrinsic,
                                           ignore_outside=False)
        elif output_type == "crop":
            new_img = crop_box_from_img(img, bbox, best_view_extrinsic_c2w, best_view_intrinsic)
            if new_img is None:
                file_name = str(object_id).zfill(3) + '_' + object_type + '_placeholder.txt'
                with open(os.path.join(output_dir, file_name), 'w') as f:
                    f.write('Object cannot be cropped properly.')
                continue
        cv2.imwrite(img_out_path, new_img)




if __name__ == '__main__':

    # scene_id = '3rscan0041'
    scene_id = 'scene0000_00'
    # scene_id = '1mp3d_0000_region0'

    # only load scene 0

    if os.path.exists(f"./{scene_id}/example.npy"):

        anno = np.load(f"./{scene_id}/example.npy", allow_pickle=True).item()
    else:

        anno = read_annotation_pickles('example_data/embodiedscan_infos_train_full.pkl')[f'{scene_id}']
        np.save(f"./{scene_id}/example.npy", anno)


    bboxes = anno['bboxes']
    object_ids = anno['object_ids']
    object_types = anno['object_types']
    visible_view_object_dict = anno['visible_view_object_dict']
    extrinsics_c2w = anno['extrinsics_c2w']
    axis_align_matrix = anno['axis_align_matrix']
    intrinsics = anno['intrinsics']
    depth_intrinsics = anno['depth_intrinsics']
    image_paths = anno['image_paths']
    dataset, _, scene_id, _ = image_paths[0].split('.')[0].split('/')

    blurry_image_ids_path =f'./{scene_id}/anno_lang/blurry_image_ids.json'


    output_dir = f'{scene_id}/anno_lang/painted_images'
    # os.makedirs(output_dir, exist_ok=True)

    real_image_paths =[f'{scene_id}/posed_images'+path.split(scene_id)[1] for path in image_paths]

    paint_object_pictures(bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix,
                          intrinsics, depth_intrinsics, real_image_paths, blurry_image_ids_path, output_dir,
                          output_type="paint")
