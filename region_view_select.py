import numpy as np
import os
import cv2
import copy
from tqdm import tqdm
from scipy.spatial import ConvexHull
import scipy
from shapely.geometry import Polygon
from utils_read import read_extrinsic, read_extrinsic_dir, read_intrinsic, read_depth_map, read_bboxes_json, load_json, reverse_multi2multi_mapping, read_annotation_pickle, EXCLUDED_OBJECTS
from utils_3d import check_bboxes_visibility, check_point_visibility, interpolate_bbox_points
from utils_read import read_annotation_pickle
import shutil
import json
from region_matching import get_data,process_data
from utils_vis import get_9dof_boxes, draw_box3d_on_img, get_color_map, crop_box_from_img
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

    bboxes = get_9dof_boxes(np.array(bboxes), mode='zxy', colors=(0, 0, 192))
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

def paint_group_pictures(group_dict,bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix,
                          intrinsics,image_paths, blurry_image_ids_path, output_dir, need_not_to_draw_list = ['wall']):
    '''given object_ids list, return fewer views possible to contain all of them and show on the images'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    bboxes = get_9dof_boxes(np.array(bboxes), mode='zxy', colors=(0, 0, 192))
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
    depth_maps = [read_depth_map(path) for path in depth_map_paths]

    # 1.delete the objects which don't have views in the dict file
    delete_secne_object = {}
    for group_name in group_dict.keys():
        delete_secne_object[group_name] = []
    for group_name in group_dict.keys():
        filter_list = []
        for object_id in group_dict[group_name]:
            if object_id in visible_object_view_dict.keys() and len(visible_object_view_dict[object_id])>0 and object_types[list(object_ids).index(object_id)] not in need_not_to_draw_list:
                filter_list.append(object_id)
            else:
                delete_secne_object[group_name].append(object_id)

        group_dict[group_name] = filter_list




    # check all the objects and find some acceptable views

    check_object_list = []


    for group_name in group_dict.keys():
        group_object_list = group_dict[group_name]
        for object_id in group_object_list:
            if object_id not in check_object_list:
                check_object_list.append(object_id)

    process_visible_object_view_dict =copy.deepcopy(visible_object_view_dict)

    for select_id in tqdm(check_object_list):
        select_views = visible_object_view_dict[select_id]
        process_visible_object_view_dict[select_id] = get_acceptable_views(select_id,select_views, bboxes, object_ids, extrinsics_c2w, view_ids,intrinsics, depth_intrinsics, depth_maps)
    process_visible_view_object_dict =reverse_multi2multi_mapping(process_visible_object_view_dict)

    for group_name in group_dict.keys():
        group_object_list = group_dict[group_name]
        filter_list = []
        for object_id in group_object_list:
            if len(process_visible_object_view_dict[object_id])>0:
                filter_list.append(object_id)
            else:
                delete_secne_object[group_name].append(object_id)

        group_dict[group_name] = filter_list

    # 3. find the fewer views containing all left objects and display

    for group_name in group_dict.keys():
        group_object_list = group_dict[group_name]
        select_views = get_fewer_view_contain_objects(process_visible_view_object_dict, process_visible_object_view_dict,group_object_list)

        _paint_group_pictures((select_views==[]),delete_secne_object[group_name],select_views,process_visible_view_object_dict, group_name, group_object_list, bboxes, object_ids, object_types,view_ids,
                              extrinsics_c2w,
                              intrinsics, color_map, image_dir, output_dir)
def _paint_group_pictures(empty,delte_object,select_views,process_visible_view_object_dict,group_name,group_object_list,bboxes, object_ids, object_types,view_ids, extrinsics_c2w,
                           intrinsics, color_map, image_dir, output_dir,
                           skip_existing=True):

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



    print('these are deleted:',delte_object)
    os.makedirs(output_dir + '/' + group_name, exist_ok=True)
    with open(output_dir+'/'+group_name+'/delete.txt','w') as f:
        f.write(str(delte_object))

    if empty:
        print("No object can be seen there in the {}".format(group_name))
        return



    for view_id in select_views:
        img_in_path = os.path.join(image_dir, view_id + '.jpg')

        img_out_path = os.path.join(output_dir+'/'+group_name+'/', view_id + '.jpg')
        img = cv2.imread(img_in_path)
        if img is None:
            # print(f"Image {img_in_path} not found, skipping object {object_id}: {object_type}")
            continue

        new_img = img


        for _id in group_object_list:

            if _id not in process_visible_view_object_dict[view_id]:
                continue

            i = list(object_ids).index(_id)
            bbox, object_id, object_type = bboxes[i], object_ids[i], object_types[i]


            color = color_map.get(object_type, (0, 0, 192))
            label = str(object_id) + ' ' + object_type
            new_img, _ = draw_box3d_on_img(img, bbox, color, label, extrinsics_c2w[view_ids.index(view_id)], intrinsics[view_ids.index(view_id)],
                                               ignore_outside=False)

        cv2.imwrite(img_out_path, new_img)

def get_acceptable_views(select_id,select_views, bboxes, object_ids,  extrinsics_c2w, view_ids,intrinsics, depth_intrinsics, depth_maps):

    '''get the acceptable views list for every object'''
    if len(np.array(intrinsics).shape) == 2:
        intrinsics = np.tile(intrinsics, (len(view_ids), 1, 1))
    if len(np.array(depth_intrinsics).shape) == 2:
        depth_intrinsics = np.tile(depth_intrinsics, (len(view_ids), 1, 1))
    id_ = list(object_ids).index(select_id)
    o3d_bbox = bboxes[id_]

    box_center = o3d_bbox.get_center()
    box_center = np.array([box_center[0], box_center[1], box_center[2], 1])
    points = interpolate_bbox_points(o3d_bbox, granularity=0.02)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)

    areas = []
    centerness = []

    n = 10000
    area_ratio_threshold = 0.4


    for view_id in select_views:
        id_1 = view_ids.index(view_id)
        selected_extrinsics_c2w = extrinsics_c2w[id_1]
        selected_depth_intrinsic = depth_intrinsics[id_1]
        selected_depth_map = depth_maps[id_1]
        height, width = selected_depth_map.shape


        pts = selected_depth_intrinsic @ np.linalg.inv(selected_extrinsics_c2w) @ points.T  # shape (4, n)
        xs, ys, zs = pts[0, :], pts[1, :], pts[2, :]
        if zs.min() < 1e-6:
            areas.append(0)
            centerness.append(0)
            continue

        xs, ys = xs / zs, ys / zs
        visible_indices = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys, zs = xs[visible_indices], ys[visible_indices], zs[visible_indices]
        xs, ys = xs.astype(int), ys.astype(int)
        visible_indices = selected_depth_map[ys, xs] > zs
        xs, ys = xs[visible_indices], ys[visible_indices]
        pts = np.stack([xs, ys], axis=-1)
        pts = np.unique(pts, axis=0)
        if len(pts) <= 2:
            areas.append(0)
            centerness.append(0)
            continue

        areas.append(_compute_area(pts))
        # now consider for centerness
        center_pos = selected_depth_intrinsic @ np.linalg.inv(selected_extrinsics_c2w) @ box_center  # shape (4,)
        cx, cy = center_pos[:2] / center_pos[2]


        if width / n <= cx and cx <= (n - 1) * width / n and height / n <= cy and cy <= (n - 1) * height / n:
            centerness.append(1)
        else:
            centerness.append(0)


    # Once
    filter_view = [True]*len(select_views)

    for _index in range(len(areas)):
        if areas[_index]==0:
            filter_view[_index]=False
    select_out1 = []
    for _index in range(len(areas)):
        if filter_view[_index]:
            select_out1.append(select_views[_index])
    #Twice

    max_area = np.max(areas)
    for _index in range(len(areas)):
        if centerness[_index]==0 or areas[_index]<200: #area_ratio_threshold*max_area:
            filter_view[_index]=False
    select_out2 = []
    for _index in range(len(areas)):
        if filter_view[_index]:
            select_out2.append(select_views[_index])
    if len(select_out2)<1:

        return select_out1

    return select_out2




def get_fewer_view_contain_objects(visible_view_object_dict,visible_object_view_dict, group_object_list):
    '''
    return fewer views to contain all the objects in the groups
    '''

    def set_sub(a, b):
        return list(set(a) - set(b))
    contain_view_list = []
    visible_view_object_dict_ = {}

    for object_id in group_object_list:
        if len(visible_object_view_dict[object_id])<1:
            return []

    for view_id in visible_view_object_dict:
        visible_view_object_dict_[view_id] = []
        for object_id in visible_view_object_dict[view_id]:
            if object_id in group_object_list:
                visible_view_object_dict_[view_id].append(object_id)

    del_keys = []
    for view_id in visible_view_object_dict_:
        if len(visible_view_object_dict_[view_id])<1:
            del_keys.append(view_id)
    for del_key in del_keys:
        visible_view_object_dict_.pop(del_key)

    while len(group_object_list)>0:
        max_num = 0
        for view_id in visible_view_object_dict_:
            if len(visible_view_object_dict_[view_id])>max_num:
                max_num = len(visible_view_object_dict_[view_id])
                max_view_id = view_id
        contain_view_list.append(max_view_id)
        filter_object_list = visible_view_object_dict_[max_view_id]
        group_object_list = set_sub(group_object_list,filter_object_list)
        for view_id in visible_view_object_dict_:
            visible_view_object_dict_[view_id] = set_sub(visible_view_object_dict_[view_id],filter_object_list)
        del_keys = []
        for view_id in visible_view_object_dict_:
            if len(visible_view_object_dict_[view_id]) < 1:
                del_keys.append(view_id)
        for del_key in del_keys:
            visible_view_object_dict_.pop(del_key)

    return contain_view_list



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
        # if show:
        #
        #
        #     import cv2
        #     depth_map[pts[:, 1], pts[:, 0]] = 255
        #     cv2.imwrite('sss_{}.png'.format(i), depth_map)
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
        # if object_id==150:
        #     print(visible_views)

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

    # 下面的code是针对一个scene，多个场景的情形类似即可
    # example.npy是pkl文件中的scene00

    scene_name = 'point_cloud_top_view'

    input_annotation_path = f"example_data/anno_lang/region_anno/{scene_name}.png.txt"  # The result of manual annotation

    # only load scene 0

    if os.path.exists("example_data/example.npy"):

        anno = np.load("example_data/example.npy", allow_pickle=True).item()
    else:

        anno = read_annotation_pickle('example_data/embodiedscan_infos_train_full.pkl')['scene0000_00']
        np.save('example_data/example.npy', anno)


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

    blurry_image_ids_path ='example_data/anno_lang/blurry_image_ids.json'


    output_dir = f'example_data/anno_lang/{scene_name}'
    os.makedirs(output_dir, exist_ok=True)

    real_image_paths =['example_data/posed_images'+path[-10:] for path in image_paths]


    # map 2d to 3d
    point_cloud_path = "example_data/lidar/main.pcd"
    min_x, min_y, region_with_label = get_data(input_annotation_path, point_cloud_path)
    region_with_label = process_data(region_with_label, object_ids, bboxes,min_x, min_y)

    group_dict = {}

    for region in region_with_label:
        group_dict[str(region['id'])+'_'+region['label']]=region['object_ids']
    print(group_dict)
    paint_group_pictures(group_dict, bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w,axis_align_matrix,intrinsics, real_image_paths, blurry_image_ids_path, output_dir)
