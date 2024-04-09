import numpy as np
import json
import os
import cv2
from tqdm import tqdm

EXCLUDED_OBJECTS = ["wall", "ceiling", "floor"]


def reverse_multi2multi_mapping(mapping):
    """
    Args:
        mapping: dict in format key1:[value1, value2], key2:[value2, value3]
    Returns:
        mapping: dict in format value1:[key1], value2:[key1, key2], value3:[key2]
    """
    output = {}
    possible_values = []
    for key, values in mapping.items():
        for value in values:
            possible_values.append(value)
    possible_values = list(set(possible_values))
    for value in possible_values:
        output[value] = []
    for key, values in mapping.items():
        for value in values:
            output[value].append(key)
    return output


def reverse_121_mapping(mapping):
    """
    Reverse a 1-to-1 mapping.
    Args:
        mapping: dict in format key1:value1, key2:value2
    Returns:
        mapping: dict in format value1:key1, value2:key2
    """
    return {v: k for k, v in mapping.items()}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_extrinsic_dir(directory):
    """
    Returns:
        extrinsics: numpy array of extrinsic matrices, shape (N, 4, 4)
        ids: list of ids (str) of matrix files.
    """
    extrinsics = []
    ids = []
    for file in os.listdir(directory):
        if file.endswith(".txt") or file.endswith(".npy"):
            if file.startswith("depth_intrinsic") or file.startswith("intrinsic"):
                continue
            path = os.path.join(directory, file)
            extrinsics.append(read_extrinsic(path))
            path = path.replace("\\", "/")
            ids.append(file.split(".")[0])
    return extrinsics, ids


def _pad_extrinsic(mat):
    """
    transforms the extrinsic matrix to the 4x4 form
    """
    mat = np.array(mat)
    if mat.shape == (3, 4):
        mat = np.vstack((mat, [0, 0, 0, 1]))
    elif mat.shape != (4, 4):
        raise ValueError("Invalid shape of matrix.")
    return mat


def read_extrinsic(path):
    """
    returns a 4x4 numpy array of intrinsic matrix
    """
    if path.endswith(".txt"):
        mat = np.loadtxt(path)
        return _pad_extrinsic(mat)
    elif path.endswith(".npy"):
        mat = np.load(path)
        return _pad_extrinsic(mat)
    else:
        raise ValueError("Invalid file extension.")


def _read_intrinsic_mp3d(path):
    a = np.loadtxt(path)
    intrinsic = np.identity(4, dtype=float)
    intrinsic[0][0] = a[2]  # fx
    intrinsic[1][1] = a[3]  # fy
    intrinsic[0][2] = a[4]  # cx
    intrinsic[1][2] = a[5]  # cy
    # a[0], a[1] are the width and height of the image
    return intrinsic


def _read_intrinsic_scannet(path):
    intrinsic = np.loadtxt(path)
    return intrinsic


def read_intrinsic(path, mode="scannet"):
    """
    Reads intrinsic matrix from file.
    Returns:
        extended intrinsic of shape (4, 4)
    """
    if mode == "scannet":
        return _read_intrinsic_scannet(path)
    elif mode == "mp3d":
        return _read_intrinsic_mp3d(path)
    else:
        raise ValueError("Invalid mode {}.".format(mode)) 

def _read_axis_align_matrix_scannet(path):
    with open(path, 'r') as file:
        first_line = file.readline()
    vals = first_line.strip().split(' ')[2:]
    vals = np.array(vals, dtype=np.float64)
    output = vals.reshape(4, 4)
    return output

def read_axis_align_matrix(path, mode):
    if mode == "scannet":
        return _read_axis_align_matrix_scannet(path)
    else:
        raise ValueError("Invalid mode {}.".format(mode)) 

def read_depth_map(path):
    """
    Reads depth map from file.
    Returns:
        depth: numpy array of depth values, shape (H, W)
    """
    if "3rscan" in path:
        path = path[:-4] + ".pgm"
    depth_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Cannot read file {path}")
    depth_map = depth_map/1000.0 # '/=' does not work. Interesting.
    if "matterport" in path or "mp3d" in path:
        depth_map /= 4.0  # for matterport, depth should be divided by 4000
    return depth_map


def read_bboxes_json(path, return_id=False, return_type=False):
    """
    Returns:
        boxes: numpy array of bounding boxes, shape (M, 9): xyz, lwh, ypr
        ids: (optional) numpy array of obj ids, shape (M,)
        types: (optional) list of strings, each string is a type of object
    """
    with open(path, "r") as f:
        bboxes_json = json.load(f)
    boxes = []
    ids = []
    types = []
    for i in range(len(bboxes_json)):
        if bboxes_json[i]["obj_type"] in EXCLUDED_OBJECTS:
            continue
        box = bboxes_json[i]["psr"]
        position = np.array(
            [box["position"]["x"], box["position"]["y"], box["position"]["z"]]
        )
        size = np.array([box["scale"]["x"], box["scale"]["y"], box["scale"]["z"]])
        euler_angles = np.array(
            [box["rotation"]["x"], box["rotation"]["y"], box["rotation"]["z"]]
        )
        boxes.append(np.concatenate([position, size, euler_angles]))
        ids.append(int(bboxes_json[i]["obj_id"]))
        types.append(bboxes_json[i]["obj_type"])
    boxes = np.array(boxes)
    if return_id and return_type:
        ids = np.array(ids)
        return boxes, ids, types
    if return_id:
        ids = np.array(ids)
        return boxes, ids
    if return_type:
        return boxes, types
    return boxes


def read_annotation_pickle(path):
    """
    Returns: A dictionary. Format. scene_id : (bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, image_paths)
    bboxes: numpy array of bounding boxes, shape (N, 9): xyz, lwh, ypr
    object_ids: numpy array of obj ids, shape (N,)
    object_types: list of strings, each string is a type of object
    visible_view_object_dict: a dictionary {view_id: visible_instance_ids}
    extrinsics_c2w: a list of 4x4 matrices, each matrix is the extrinsic matrix of a view
    axis_align_matrix: a 4x4 matrix, the axis-aligned matrix of the scene
    intrinsics: a list of 4x4 matrices, each matrix is the intrinsic matrix of a view
    image_paths: a list of strings, each string is the path of an image in the scene
    """
    with open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
    metainfo = data["metainfo"]
    object_type_to_int = metainfo["categories"]
    object_int_to_type = {v: k for k, v in object_type_to_int.items()}
    datalist = data["data_list"]
    output_data = {}
    pbar = tqdm(range(len(datalist)))
    for scene_idx in pbar:
        images = datalist[scene_idx]["images"]
        intrinsic = datalist[scene_idx].get("cam2img", None)  # a 4x4 matrix
        missing_intrinsic = False
        if intrinsic is None:
            missing_intrinsic = True  # each view has different intrinsic for mp3d
        depth_intrinsic = datalist[scene_idx].get(
            "cam2depth", None
        )  # a 4x4 matrix, for 3rscan
        if depth_intrinsic is None and not missing_intrinsic:
            depth_intrinsic = datalist[scene_idx][
                "depth2img"
            ]  # a 4x4 matrix, for scannet
        axis_align_matrix = datalist[scene_idx]["axis_align_matrix"]  # a 4x4 matrix
        scene_id = images[0]["img_path"].split("/")[-2]  # str

        instances = datalist[scene_idx]["instances"]
        bboxes = []
        object_ids = []
        object_types = []
        for object_idx in range(len(instances)):
            bbox_3d = instances[object_idx]["bbox_3d"]  # list of 9 values
            bbox_label_3d = instances[object_idx]["bbox_label_3d"]  # int
            bbox_id = instances[object_idx]["bbox_id"]  # int
            object_type = object_int_to_type[bbox_label_3d]
            # if object_type in EXCLUDED_OBJECTS:
            #     continue
            object_types.append(object_type)
            bboxes.append(bbox_3d)
            object_ids.append(bbox_id)
        bboxes = np.array(bboxes)
        object_ids = np.array(object_ids)

        visible_view_object_dict = {}
        extrinsics_c2w = []
        intrinsics = []
        depth_intrinsics = []
        image_paths = []
        for image_idx in range(len(images)):
            img_path = images[image_idx]["img_path"]  # str
            extrinsic_id = img_path.split("/")[-1].split(".")[0]  # str
            cam2global = images[image_idx]["cam2global"]  # a 4x4 matrix
            if missing_intrinsic:
                intrinsic = images[image_idx]["cam2img"]
                depth_intrinsic = images[image_idx]["cam2depth"]
            visible_instance_indices = images[image_idx][
                "visible_instance_ids"
            ]  # numpy array of int
            visible_instance_ids = object_ids[visible_instance_indices]
            visible_view_object_dict[extrinsic_id] = visible_instance_ids
            extrinsics_c2w.append(cam2global)
            intrinsics.append(intrinsic)
            depth_intrinsics.append(depth_intrinsic)
            image_paths.append(img_path)

        pbar.set_description(f"Processing scene {scene_id}")
        output_data[scene_id] = {
            "bboxes": bboxes,
            "object_ids": object_ids,
            "object_types": object_types,
            "visible_view_object_dict": visible_view_object_dict,
            "extrinsics_c2w": extrinsics_c2w,
            "axis_align_matrix": axis_align_matrix,
            "intrinsics": intrinsics,
            "depth_intrinsics": depth_intrinsics,
            "image_paths": image_paths,
        }
    return output_data

def read_annotation_pickles(paths):
    """
        Read multiple annotation pickles and merge them into one dictionary.
        Args:
            paths: a list of paths to annotation pickles.
        Returns: Please refer to the return value of read_annotation_pickle()
    """
    output_data = {}
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        data = read_annotation_pickle(path)
        output_data.update(data)
    output_data = dict(sorted(output_data.items()))
    return output_data
        

if __name__ == "__main__":
    pickle_file = "./example_data/embodiedscan_infos_val_full.pkl"
    read_annotation_pickle(pickle_file)
