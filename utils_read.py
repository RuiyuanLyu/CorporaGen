import numpy as np
import json
import os
EXCLUDED_OBJECTS = ['wall', 'ceiling', 'floor']

def reverse_multi2multi_mapping(mapping):
    """
        Args:
            mapping in format key1:[value1, value2], key2:[value2, value3]
        Returns:
            mapping in format value1:[key1], value2:[key1, key2], value3:[key2]
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

def load_json(path):
    with open(path, 'r') as f:
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
        if file.endswith('.txt') or file.endswith('.npy'):
            if file.startswith('depth_intrinsic') or file.startswith('intrinsic'):
                continue
            path = os.path.join(directory, file)
            extrinsics.append(read_extrinsic(path))
            path = path.replace("\\", "/")
            ids.append(file.split('.')[0])
    return extrinsics, ids

def _pad_extrinsic(mat):
    """
        transforms the extrinsic matrix to the 4x4 form
    """
    mat = np.array(mat)
    if mat.shape == (3, 4):
        mat = np.vstack((mat, [0, 0, 0, 1]))
    elif mat.shape != (4, 4):
        raise ValueError('Invalid shape of matrix.')
    return mat

def read_extrinsic(path):
    """
        returns a 4x4 numpy array of intrinsic matrix
    """
    if path.endswith('.txt'):
        mat = np.loadtxt(path)
        return _pad_extrinsic(mat)
    elif path.endswith('.npy'):
        mat = np.load(path)
        return _pad_extrinsic(mat)
    else:
        raise ValueError('Invalid file extension.')

def _read_mp3d_intrinsic(path):
    a = np.loadtxt(path)
    intrinsic = np.identity(4, dtype=float)
    intrinsic[0][0] = a[2]  # fx
    intrinsic[1][1] = a[3]  # fy
    intrinsic[0][2] = a[4]  # cx
    intrinsic[1][2] = a[5]  # cy
    # a[0], a[1] are the width and height of the image
    return intrinsic

def _read_scannet_intrinsic(path):
    intrinsic =  np.loadtxt(path)
    return intrinsic

def read_intrinsic(path, mode='scannet'):
    """
        Reads intrinsic matrix from file.
        Returns:
            extended intrinsic of shape (4, 4)
    """
    if mode =='scannet':
        return _read_scannet_intrinsic(path)
    elif mode =='mp3d':
        return _read_mp3d_intrinsic(path)
    else:
        raise ValueError('Invalid mode.')

def read_bboxes_json(path, return_id=False, return_type=False):
    """
        Returns:
            boxes: numpy array of bounding boxes, shape (M, 9): xyz, lwh, ypr
            ids: (optional) numpy array of obj ids, shape (M,)
            types: (optional) list of strings, each string is a type of object
    """
    with open(path, 'r') as f:
        bboxes_json = json.load(f)
    boxes = []
    ids = []
    types = []
    for i in range(len(bboxes_json)):
        if bboxes_json[i]['obj_type'] in EXCLUDED_OBJECTS:
            continue
        box = bboxes_json[i]["psr"]
        position = np.array([box['position']['x'], box['position']['y'], box['position']['z']])
        size =  np.array([box['scale']['x'], box['scale']['y'], box['scale']['z']])
        euler_angles = np.array([box['rotation']['x'], box['rotation']['y'], box['rotation']['z']])
        boxes.append(np.concatenate([position, size, euler_angles]))
        ids.append(int(bboxes_json[i]['obj_id']))
        types.append(bboxes_json[i]['obj_type'])
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

